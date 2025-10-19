package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.Threads;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.LinearRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.optimizer.GradientDescentOptimizer;
import net.tvburger.jdl.model.training.optimizer.steps.VanillaGradientDescent;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.Map;

public class OptimizationLearningRateSize {
/*
    public static void main(String[] args) {
        showForOverall();
    }

    private static void showForAnyStepSize() {
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 1.39f, 2));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 1.40f, 2));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 1.01f, 8));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 1.02f, 8));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.98f, 9));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.99f, 9));
    }

    private static void showForOverall() {
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 1000, 0.99f, 9));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.97f, 9));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.98f, 9));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.99f, 9));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.97f, 8));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.98f, 8));
        Threads.runAsynchronously(() -> showForNumberType(JavaNumberTypeSupport.DOUBLE, 100, 0.99f, 8));
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, int epochs, float learningRate, int m) {
        Plot mrePlot = new Plot("RME " + " m=" + m + ", type=" + typeSupport.name());
        mrePlot.display();

        Plot fitPlot = new Plot("Fit " + " m=" + m + ", type=" + typeSupport.name());
        fitPlot.display();

        System.out.println("Type: " + typeSupport.name() + ", Learning rate: " + learningRate);
        showForNumberType(typeSupport, epochs, learningRate, m, mrePlot, fitPlot);
        Threads.sleepSilently(1_000);
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, int epochs, float learningRate, int m, Plot mrePlot, Plot fitPlot) {
        LinearRegression<N> regression = createClosedSolutionFitting(typeSupport);
        fitPlot.plotDataSet(regression.getTrainSet(), "Train Set");
        fitPlot.redraw();
        VanillaGradientDescentOptimizer<N> optimizer = (VanillaGradientDescentOptimizer<N>) regression.getOptimizer();
        optimizer.setLearningRate(learningRate);
        optimizer.setEpochs(epochs);
        optimizer.setEpochCompletionListener(IterativeOptimizer.sample(100, rmePlotter(mrePlot, fitPlot, regression, learningRate)));
        regression.fitComplexity(m);
    }

    private static synchronized <N extends Number> VanillaGradientDescentOptimizer.EpochCompletionListener<N> rmePlotter(Plot mrePlot, Plot fitPlot, LinearRegression<N> regression, float learningRate) {
        return (epoch, model, trainSet, optimizer) -> {
            fitPlot.plotTargetOutput(model, "Fit (" + learningRate + ")");
            fitPlot.redraw();
            Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
            mrePlot.addToSeries("Train RME (" + learningRate + ")", new float[]{epoch}, new float[]{currentRmes.left()});
            for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                mrePlot.addToSeries(currentRmeEntry.getKey() + " RME (" + learningRate + ")", new float[]{epoch}, new float[]{currentRmeEntry.getValue()});
            }
            mrePlot.redraw();
        };
    }

    private static <N extends Number> LinearRegression<N> createClosedSolutionFitting(JavaNumberTypeSupport<N> typeSupport) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator);
    }

    public static <N extends Number> LinearRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(1.0f);
        DataSet<N> trainSet = dataSetGenerator.load(10);
        DataSet<N> testSet = dataSetGenerator.load(10);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, new GradientDescentOptimizer<>(new VanillaGradientDescent<>(basisFunctionGenerator.getCurrentNumberType().valueOf(0.1f))), Regimes.batch());
    }
*/
}
