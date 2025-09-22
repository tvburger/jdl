package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.ClosedSolutionRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.model.DataSet;

import java.util.LinkedHashMap;
import java.util.Map;

public class ModelComplexity {

    public static void main(String[] args) {
        showForNumberType(JavaNumberTypeSupport.RATIONAL_BIGINT);
    }

    private static <N extends Number> ClosedSolutionRegression<N> createClosedSolutionFitting(JavaNumberTypeSupport<N> typeSupport) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator);
    }

    public static <N extends Number> ClosedSolutionRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(1.0f);
        DataSet<N> trainSet = dataSetGenerator.load(10);
        DataSet<N> testSet = dataSetGenerator.load(1000);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new ClosedSolutionRegression<>(dataSetGenerator, basisFunctionGenerator, trainSet, testSets);
    }

    public static void showForNumberType(JavaNumberTypeSupport<? extends Number> typeSupport) {
        ClosedSolutionRegression<?> estimation = createClosedSolutionFitting(typeSupport);
        for (int m = 0; m < 10; m++) {
            if (m != 0 && m % 2 == 0) {
                continue; // skip M = 2, 4, 6, etc.
            }
            estimation.fitComplexity(m);
        }

        estimation.showErrorPlot();
        estimation.showWeightsPlot();

        estimation.setShowTrainSet(true);
        estimation.setShowTargetFit(true);
        estimation.showFitPlot();
    }

}
