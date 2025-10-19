package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.SimpleHolder;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
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
import net.tvburger.jdl.model.training.optimizer.steps.*;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.EpochRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;
import net.tvburger.jdl.plots.Plot;

import java.util.*;

public class NewOptimizers {

    public static boolean SHOW_FIT = false;

    @SuppressWarnings("unchecked")
    public static <N extends Number> void main(String[] args) {
        JavaNumberTypeSupport<N> typeSupport = (JavaNumberTypeSupport<N>) JavaNumberTypeSupport.DOUBLE;
        N learningRate = typeSupport.valueOf(0.1f);
        int epochs = 100_000;
        int m = 9; // model complexity

        Plot weightPlot = new Plot("Absolute sum of weights");
        weightPlot.display();

        Plot stepSizePlot = new Plot("Step size");
        stepSizePlot.setYRange(0.0f, 1.0f);
        stepSizePlot.display();

        Plot mrePlot = new Plot("Train RME");
        mrePlot.setYRange(0.0f, 1.0f);
        mrePlot.display();

        VanillaGradientDescent<N> vanilla = new VanillaGradientDescent<>(learningRate);
        AdaGrad<N> adagrad = new AdaGrad<>(learningRate);
        RMSProp<N> rmsProp = new RMSProp<>(learningRate, typeSupport.valueOf(0.9));
        Adam<N> adam = new Adam<>(learningRate, typeSupport.valueOf(0.9), typeSupport.valueOf(0.999));
        AdamW<N> adamW = new AdamW<>(learningRate, typeSupport.valueOf(0.9), typeSupport.valueOf(0.999), typeSupport.valueOf(0.001));
        ARGOptimizer<N> arg = new ARGOptimizer<>(learningRate);

        Map<String, UpdateStep<LinearCombination<N>, N>> optimizers = new LinkedHashMap<>() {
            {
                put("Vanilla", vanilla);
//                put("AdaGrad", adagrad);
//                put("RMSProp", rmsProp);
                put("Adam", adam);
                put("AdamW", adamW);
                put("ARG", arg);
            }
        };
        for (Map.Entry<String, UpdateStep<LinearCombination<N>, N>> optimizerEntry : optimizers.entrySet()) {
            showForNumberType(typeSupport, optimizerEntry.getKey(), createOptimizer(optimizerEntry.getValue()), Set.of(), epochs, m, weightPlot, mrePlot, stepSizePlot);
        }
    }

    private static <N extends Number> Optimizer<LinearBasisFunctionModel<N>, N> createOptimizer(UpdateStep<LinearCombination<N>, N> updateStep) {
        LinearBasisFunctionModelDecomposer<N> decomposer = new LinearBasisFunctionModelDecomposer<>();
        return new GradientDescentOptimizer<>(decomposer, updateStep);
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, String title, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Set<ExplicitRegularization<N>> regularizations, int epochs, int m, Plot weightPlot, Plot mrePlot, Plot stepSizePlot) {
        if (SHOW_FIT) {
            System.out.println("Showing fit for optimizer: " + title);
        }
        Plot fitPlot = new Plot("Fit " + title);
        fitPlot.setYRange(-2, 2);

        SimpleHolder<LinearRegression<N>> regressionHolder = SimpleHolder.create();
        Map<Optimizer<?, ?>, Vector<N>> previousParameters = new HashMap<>();
        ChainedRegime regime = Regimes.epochs(epochs, EpochRegime.sample(1000, new EpochRegime.EpochCompletionListener() {
            @SuppressWarnings("unchecked")
            @Override
            public <M extends Number> void epochCompleted(EpochRegime epochRegime, int currentEpoch, TrainableFunction<M> model, DataSet<M> trainingSet, Optimizer<? extends TrainableFunction<M>, M> optimizer) {
                LinearBasisFunctionModel<N> linearBasisFunctionModel = (LinearBasisFunctionModel<N>) model;
                Pair<Float, Map<String, Float>> currentRmes = regressionHolder.get().calculateRMEs(linearBasisFunctionModel);
                mrePlot.addToSeries(title, new float[]{currentEpoch}, new float[]{currentRmes.left()});
                mrePlot.redraw();
                weightPlot.addToSeries(title, new float[]{currentEpoch}, new float[]{(float) (Arrays.stream(model.getParameters()).mapToDouble(p -> model.getCurrentNumberType().absolute(p).doubleValue()).sum())});
                weightPlot.redraw();
                if (SHOW_FIT) {
                    fitPlot.plotTargetOutput(linearBasisFunctionModel, "Fit");
                }
                fitPlot.redraw();
                Vector<N> oldParameters = previousParameters.get(optimizer);
                Vector<N> newParameters = Vectors.of(linearBasisFunctionModel.getCurrentNumberType(), linearBasisFunctionModel.getParameters()).transpose();
                if (oldParameters != null) {
                    N magnitude = newParameters.subtract(oldParameters).norm();
                    stepSizePlot.addToSeries(title, new float[]{currentEpoch}, new float[]{magnitude.floatValue()});
                    stepSizePlot.redraw();
                }
                previousParameters.put(optimizer, newParameters);
                if (currentEpoch == epochs) {
                    System.out.println(title + ": " + Arrays.toString(model.getParameters()));
                }
            }
        })).batch();
        LinearRegression<N> regression = createIterativeTrainedRegression(typeSupport, optimizer, regime);
        regressionHolder.set(regression);

        if (SHOW_FIT) {
            fitPlot.plotDataSet(regression.getTrainSet(), "Train Set");
            fitPlot.display();
        }

        regression.fitComplexity(m);
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
