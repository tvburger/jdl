package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.RegularizedClosedSolutionRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.model.DataSet;

import java.util.LinkedHashMap;
import java.util.Map;

public class ModelRegularization {

    public static void main(String[] args) {
        showForNumberType(JavaNumberTypeSupport.DOUBLE);
    }

    private static <N extends Number> RegularizedClosedSolutionRegression<N> createClosedSolutionFitting(JavaNumberTypeSupport<N> typeSupport) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator);
    }

    public static <N extends Number> RegularizedClosedSolutionRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(0.5f);
        DataSet<N> trainSet = dataSetGenerator.load(10);
        DataSet<N> testSet = dataSetGenerator.load(1000);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new RegularizedClosedSolutionRegression<>(dataSetGenerator, basisFunctionGenerator, trainSet, testSets);
    }

    public static void showForNumberType(JavaNumberTypeSupport<? extends Number> typeSupport) {
        RegularizedClosedSolutionRegression<?> estimation = createClosedSolutionFitting(typeSupport);
        estimation.DEBUG_OUTPUT = true;

//        for (float i : new float[]{0.0000001f, 0.000001f, 0.00001f, 0.005f, 0.01f, 0.05f, 0.1f, 0.5f, 1.0f}) {
        for (float i : new float[]{1e-15f, 1e-12f, 1e-10f, 1e-9f, 1e-8f, 1e-7f, 1e-5f, 1e-4f, 1e-3f, 0.005f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 1.0f}) {
            estimation.setLambda(i);
            estimation.fitComplexity(7);
        }

        estimation.showErrorPlot();
        estimation.showWeightsPlot();

        estimation.setShowTrainSet(true);
        estimation.setShowTargetFit(true);
        estimation.showFitPlot();
    }

}
