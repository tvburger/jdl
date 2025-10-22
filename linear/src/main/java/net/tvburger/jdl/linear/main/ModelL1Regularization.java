package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.SimpleHolder;
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
import net.tvburger.jdl.model.training.regularization.RegularizationFactory;
import net.tvburger.jdl.model.training.regularization.Regularizations;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public class ModelL1Regularization {

    @SuppressWarnings("unchecked")
    public static <N extends Number> void main(String[] args) {
        JavaNumberTypeSupport<N> numberTypeSupport = (JavaNumberTypeSupport<N>) JavaNumberTypeSupport.DOUBLE;
        RegularizationFactory<N> regularizationFactory = Regularizations.getFactory(numberTypeSupport);
        N learningRate = numberTypeSupport.valueOf(0.5f);
        int epochs = 100_000;
        int m = 9; // model complexity

        VanillaGradientDescent<N> vanilla = new VanillaGradientDescent<>(learningRate);
        ExplicitRegularization<N> elasticNet = regularizationFactory.createElasticNet(0.001, 0.0001);
        ExplicitRegularization<N> l1 = regularizationFactory.createLASSO(0.001);
        ExplicitRegularization<N> l2 = regularizationFactory.createRidge(0.0001);

        Plot mrePlot = new Plot("RME");
        mrePlot.display();
        Plot fitPlot = new Plot("Fit");
        fitPlot.setYRange(-2, 2);
        fitPlot.display();
        Plot weightPlot = new Plot("Absolute sum of weights");
        weightPlot.display();

        Map<String, Pair<UpdateStep<LinearCombination<N>, N>, Set<ExplicitRegularization<N>>>> tests = new LinkedHashMap<>() {
            {
                put("Vanilla", Pair.of(vanilla, Set.of()));
                put("L1", Pair.of(vanilla, Set.of(l1)));
                put("L2", Pair.of(vanilla, Set.of(l2)));
                put("L1 + L2", Pair.of(vanilla, Set.of(elasticNet)));
            }
        };
        for (Map.Entry<String, Pair<UpdateStep<LinearCombination<N>, N>, Set<ExplicitRegularization<N>>>> optimizerEntry : tests.entrySet()) {
            showForNumberType(numberTypeSupport, optimizerEntry.getKey(), createOptimizer(optimizerEntry.getValue().left()), optimizerEntry.getValue().right(), epochs, m, mrePlot, fitPlot, weightPlot);
        }
    }

    private static <N extends Number> Optimizer<LinearBasisFunctionModel<N>, N> createOptimizer(UpdateStep<LinearCombination<N>, N> updateStep) {
        LinearBasisFunctionModelDecomposer<N> decomposer = new LinearBasisFunctionModelDecomposer<>();
        return new GradientDescentOptimizer<>(decomposer, updateStep);
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, String title, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Set<ExplicitRegularization<N>> regularizations, int epochs, int m, Plot mrePlot, Plot fitPlot, Plot weightPlot) {
        SimpleHolder<LinearRegression<N>> regressionHolder = SimpleHolder.create();

        ChainedRegime regime = Regimes.epochs(epochs, EpochRegime.sample(250, new EpochRegime.EpochCompletionListener() {
            @SuppressWarnings("unchecked")
            @Override
            public <M extends Number> void epochCompleted(EpochRegime epochRegime, int currentEpoch, TrainableFunction<M> model, DataSet<M> trainingSet, Optimizer<? extends TrainableFunction<M>, M> optimizer) {
                LinearBasisFunctionModel<N> linearBasisFunctionModel = (LinearBasisFunctionModel<N>) model;
                Pair<Float, Map<String, Float>> currentRmes = regressionHolder.get().calculateRMEs(linearBasisFunctionModel);
                mrePlot.addToSeries("Train " + title, new float[]{currentEpoch}, new float[]{currentRmes.left()});
                for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                    mrePlot.addToSeries(currentRmeEntry.getKey() + " " + title, new float[]{currentEpoch}, new float[]{currentRmeEntry.getValue()});
                }
                mrePlot.redraw();
                weightPlot.addToSeries(title, new float[]{currentEpoch}, new float[]{(float) (Array.stream(model.getParameters()).mapToDouble(p -> model.getNumberTypeSupport().absolute(p).doubleValue()).sum())});
                weightPlot.redraw();
                fitPlot.plotTargetOutput(linearBasisFunctionModel, title);
                fitPlot.redraw();
                if (currentEpoch == epochs) {
                    System.out.println(title + ": " + Array.toString(model.getParameters()));
                }
            }
        })).batch();
        LinearRegression<N> regression = createIterativeTrainedRegression(typeSupport, optimizer, regime);
        regressionHolder.set(regression);

        if (!fitPlot.hasSeries("Train Set")) {
            fitPlot.plotDataSet(regression.getTrainSet(), "Train Set");
        }

        regression.fitComplexity(m, regularizations);
    }

    private static <N extends Number> LinearRegression<N> createIterativeTrainedRegression(JavaNumberTypeSupport<N> typeSupport, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Regime regime) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator, optimizer, regime);
    }

    public static <N extends Number> LinearRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Regime regime) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(1.0f);
        DataSet<N> trainSet = dataSetGenerator.load(5);
        DataSet<N> testSet = dataSetGenerator.load(10);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, optimizer, regime);
    }

}
