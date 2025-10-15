package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.Threads;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.LinearRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.linear.optimizer.VanillaGradientDescentOptimizer;
import net.tvburger.jdl.linear.optimizer.IterativeOptimizer;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.Map;

public class OptimizationModelComplexity {

    public static void main(String[] args) {
        showForNumberType(JavaNumberTypeSupport.DOUBLE, 10_000_000, 0.97f);
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, int epochs, float learningRate) {
        Plot weightPlot = new Plot("Average absolute weights");
        weightPlot.display();

        Plot weight2Plot = new Plot("Sum of absolute weights");
//        weight2Plot.display();

        Plot mrePlot = new Plot("RME " + " type=" + typeSupport.name());
        mrePlot.display();

        System.out.println("Type: " + typeSupport.name() + ", Learning rate: " + learningRate);
        for (int m = 5; m <= 12; m = m + 2) {
            showForNumberType(typeSupport, epochs, learningRate, m, mrePlot, weightPlot, weight2Plot);
            Threads.sleepSilently(1_000);
        }
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, int epochs, float learningRate, int m, Plot mrePlot, Plot weightPlot, Plot weight2Plot) {
        LinearRegression<N> regression = createClosedSolutionFitting(typeSupport);
        VanillaGradientDescentOptimizer<N> optimizer = (VanillaGradientDescentOptimizer<N>) regression.getOptimizer();
        optimizer.setLearningRate(learningRate);
        optimizer.setEpochs(epochs);
        optimizer.setEpochCompletionListener(IterativeOptimizer.sample(250, rmePlotter(mrePlot, weightPlot, weight2Plot, regression, learningRate)));
        regression.fitComplexity(m);
    }

    private static synchronized <N extends Number> VanillaGradientDescentOptimizer.EpochCompletionListener<N> rmePlotter(Plot mrePlot, Plot weightPlot, Plot weight2Plot, LinearRegression<N> regression, float learningRate) {
        return (epoch, model, trainSet, optimizer) -> {
            Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
            mrePlot.addToSeries("Train RME (" + learningRate + ", " + model.getModelComplexity() + ")", new float[]{epoch}, new float[]{currentRmes.left()});
            for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                mrePlot.addToSeries(currentRmeEntry.getKey() + " RME (" + learningRate + ", " + model.getModelComplexity() + ")", new float[]{epoch}, new float[]{currentRmeEntry.getValue()});
            }
            mrePlot.redraw();

            float totalWeights = 0.0f;
            for (N parameter : model.getParameters()) {
                totalWeights += model.getCurrentNumberType().absolute(parameter).floatValue() / model.getModelComplexity();
            }
            weightPlot.addToSeries("Model (" + learningRate + ", " + model.getModelComplexity() + ")", new float[]{epoch}, new float[]{totalWeights});
            weightPlot.redraw();

//            totalWeights = 0.0f;
//            for (N parameter : model.getParameters()) {
//                totalWeights += model.getCurrentNumberType().absolute(parameter).floatValue();
//            }
//            weight2Plot.addToSeries("Model (" + learningRate + ", " + model.getModelComplexity() + ")", new float[]{epoch}, new float[]{totalWeights});
//            weight2Plot.redraw();
        };
    }

    private static <N extends Number> LinearRegression<N> createClosedSolutionFitting(JavaNumberTypeSupport<N> typeSupport) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator);
    }

    public static <N extends Number> LinearRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(0.5f);
        DataSet<N> trainSet = dataSetGenerator.load(10);
        DataSet<N> testSet = dataSetGenerator.load(10);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, new VanillaGradientDescentOptimizer<>(basisFunctionGenerator.getCurrentNumberType()));
    }

}
