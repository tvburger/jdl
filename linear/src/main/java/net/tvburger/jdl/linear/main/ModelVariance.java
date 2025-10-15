package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.Threads;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linalg.Notations;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.linear.LinearRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.linear.optimizer.L2RegularizedClosedSolutionOptimizer;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.Map;

public class ModelVariance {

    public static void main(String[] args) {
        useNumberType(JavaNumberTypeSupport.RATIONAL_BIGINT, 25, 8, 9, 0);
    }

    private static <N extends Number> void useNumberType(JavaNumberTypeSupport<N> typeSupport, int count, double regularizationLog10, int m, long sleep) {
        Plot weightsPlot = new Plot("Total Absolute Weights");
        weightsPlot.getChart().setXAxisTitle("iteration");
        weightsPlot.display();

        Plot mrePlot = new Plot("RME");
        mrePlot.getChart().setXAxisTitle("iteration");
        mrePlot.display();

        Plot fitPlot = new Plot("Variance vs Regularization");
        fitPlot.plotTargetOutput(SyntheticDataSets.sinus(typeSupport).getEstimationFunction(), "Target");
        fitPlot.setYRange(-2.0, 2.0);
        fitPlot.display();

        N lambda = typeSupport.one();
        N e = typeSupport.valueOf(10);
        for (int i = 0; i < regularizationLog10; i++) {
            lambda = typeSupport.divide(lambda, e);
        }
        for (int i = 0; i < count; i++) {
            Threads.sleepSilently(sleep);

            String name = "log10(" + Notations.LAMBDA + ") = -" + regularizationLog10;
            System.out.println(i + ": " + name);
            LinearRegression<N> regression = createClosedSolutionFitting(typeSupport);
            ((L2RegularizedClosedSolutionOptimizer<N>) regression.getOptimizer()).setLambda(lambda);
            LinearBasisFunctionModel<N> model = regression.fitComplexity(m);

            fitPlot.plotTargetOutput(model, "Regularized Fit");
            Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
            mrePlot.addToSeries("Train RME", new float[]{i}, new float[]{currentRmes.left()});
            for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                mrePlot.addToSeries(currentRmeEntry.getKey() + " RME", new float[]{i}, new float[]{currentRmeEntry.getValue()});
            }
            float totalWeights = 0.0f;
            for (int p = 0; p < model.getParameterCount(); p++) {
                totalWeights += typeSupport.absolute(model.getParameter(p)).floatValue();
            }
            weightsPlot.addToSeries("Total Weights", new float[]{i}, new float[]{totalWeights});
            fitPlot.getChart().setTitle(name);
            fitPlot.redraw();
            mrePlot.redraw();
            weightsPlot.redraw();
        }
    }

    private static <N extends Number> LinearRegression<N> createClosedSolutionFitting(JavaNumberTypeSupport<N> typeSupport) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator);
    }

    public static <N extends Number> LinearRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator) {
        dataSetGenerator.setNoiseScale(0.5f);
        DataSet<N> trainSet = dataSetGenerator.load(10);
        DataSet<N> testSet = dataSetGenerator.load(1000);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, new L2RegularizedClosedSolutionOptimizer<>(basisFunctionGenerator.getCurrentNumberType()));
    }

}
