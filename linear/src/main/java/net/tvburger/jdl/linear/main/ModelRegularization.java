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

public class ModelRegularization {

    public static void main(String[] args) {
        showForNumberType(JavaNumberTypeSupport.RATIONAL_BIGINT, 1, 20, 9, 0);
    }

    private static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, int begin, int end, int m, long sleep) {
        LinearRegression<N> regression = createClosedSolutionFitting(typeSupport);

        Plot mrePlot = new Plot("RME");
        mrePlot.display();

        Plot fitPlot = new Plot("Fit vs Regularization");
        fitPlot.setYRange(-2, 2);
        ((L2RegularizedClosedSolutionOptimizer<N>) regression.getOptimizer()).setLambda(typeSupport.zero());
        fitPlot.plotTargetOutput(regression.fitComplexity(m), "Overfitted");
        fitPlot.plotTargetOutput(SyntheticDataSets.sinus(typeSupport).getEstimationFunction(), "Target");
        fitPlot.display();

        N lambda = typeSupport.one();
        N e = typeSupport.valueOf(10);
        for (int i = 0; i < begin; i++) {
            lambda = typeSupport.divide(lambda, e);
        }
        for (int i = begin; i <= end; i++) {
            Threads.sleepSilently(sleep);

            String name = "log10(" + Notations.LAMBDA + ") = -" + i;
            System.out.println(name + " (N = " + lambda + ")");
            ((L2RegularizedClosedSolutionOptimizer<N>) regression.getOptimizer()).setLambda(lambda);
            LinearBasisFunctionModel<N> model = regression.fitComplexity(m);
            lambda = typeSupport.divide(lambda, e);

            fitPlot.plotTargetOutput(model, "Regularized Fit");
            Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
            mrePlot.addToSeries("Train RME", new float[]{-i}, new float[]{currentRmes.left()});
            for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                mrePlot.addToSeries(currentRmeEntry.getKey() + " RME", new float[]{-i}, new float[]{currentRmeEntry.getValue()});
            }
            fitPlot.getChart().setTitle(name);
            fitPlot.redraw();
            mrePlot.redraw();
        }
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
        DataSet<N> testSet = dataSetGenerator.load(1000);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, new L2RegularizedClosedSolutionOptimizer<>(basisFunctionGenerator.getCurrentNumberType()));
    }

}
