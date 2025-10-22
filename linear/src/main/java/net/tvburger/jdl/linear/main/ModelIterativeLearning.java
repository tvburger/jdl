package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.SimpleHolder;
import net.tvburger.jdl.common.utils.Threads;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.linear.LinearRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.linear.optimizer.LinearBasisFunctionModelDecomposer;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.TrainableFunction;
import net.tvburger.jdl.model.training.optimizer.GradientDescentOptimizer;
import net.tvburger.jdl.model.training.optimizer.UpdateStep;
import net.tvburger.jdl.model.training.optimizer.steps.VanillaGradientDescent;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.EpochRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public class ModelIterativeLearning {

    @SuppressWarnings("unchecked")
    public static <N extends Number> void main(String[] args) {
        JavaNumberTypeSupport<N> numberTypeSupport = (JavaNumberTypeSupport<N>) JavaNumberTypeSupport.DOUBLE;
        N learningRate = numberTypeSupport.valueOf(0.9f);
        int epochs = 10_000_000;
        int m = 9; // model complexity

        VanillaGradientDescent<N> vanilla = new VanillaGradientDescent<>(learningRate);
        showForNumberType(numberTypeSupport, "Vanilla", createOptimizer(vanilla), Set.of(), epochs, m);
    }

    private static <N extends Number> Optimizer<LinearBasisFunctionModel<N>, N> createOptimizer(UpdateStep<LinearCombination<N>, N> updateStep) {
        LinearBasisFunctionModelDecomposer<N> decomposer = new LinearBasisFunctionModelDecomposer<>();
        return new GradientDescentOptimizer<>(decomposer, updateStep);
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, String title, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Set<ExplicitRegularization<N>> regularizations, int epochs, int m) {
        Plot mrePlot = new Plot("RME " + title);
        Plot fitPlot = new Plot("Fit " + title);
        Plot weightPlot = new Plot("Absolute sum of weights " + title);
        SimpleHolder<LinearRegression<N>> regressionHolder = SimpleHolder.create();

        ChainedRegime regime = Regimes.epochs(epochs, EpochRegime.sample(250, new EpochRegime.EpochCompletionListener() {
            @SuppressWarnings("unchecked")
            @Override
            public <M extends Number> void epochCompleted(EpochRegime epochRegime, int currentEpoch, TrainableFunction<M> model, DataSet<M> trainingSet, Optimizer<? extends TrainableFunction<M>, M> optimizer) {
                LinearBasisFunctionModel<N> linearBasisFunctionModel = (LinearBasisFunctionModel<N>) model;
                Pair<Float, Map<String, Float>> currentRmes = regressionHolder.get().calculateRMEs(linearBasisFunctionModel);
                mrePlot.addToSeries("Train RME", new float[]{currentEpoch}, new float[]{currentRmes.left()});
                for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                    mrePlot.addToSeries(currentRmeEntry.getKey() + " RME", new float[]{currentEpoch}, new float[]{currentRmeEntry.getValue()});
                }
                mrePlot.redraw();
                weightPlot.addToSeries("Total Weights", new float[]{currentEpoch}, new float[]{(float) (Array.stream(model.getParameters()).mapToDouble(p -> model.getNumberTypeSupport().absolute(p).doubleValue()).sum())});
                weightPlot.redraw();
                fitPlot.plotTargetOutput(linearBasisFunctionModel, "Fit");
                fitPlot.redraw();
                if (currentEpoch == epochs) {
                    System.out.println(title + ": " + Array.toString(model.getParameters()));
                }
            }
        })).batch();
        LinearRegression<N> regression = createIterativeTrainedRegression(typeSupport, optimizer, regime);
        regressionHolder.set(regression);

        mrePlot.display();

        fitPlot.setYRange(-2, 2);
        fitPlot.plotDataSet(regression.getTrainSet(), "Train Set");
        fitPlot.display();

        weightPlot.display();

        Threads.runAsynchronously(() -> regression.fitComplexity(m, regularizations));
    }

    private static <N extends Number> LinearRegression<N> createIterativeTrainedRegression(JavaNumberTypeSupport<N> typeSupport, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Regime regime) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator, optimizer, regime);
    }

    public static <N extends Number> LinearRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Regime regime) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(0.5f);
        DataSet<N> trainSet = dataSetGenerator.load(10);
        DataSet<N> testSet = dataSetGenerator.load(10);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, optimizer, regime);
    }

}
