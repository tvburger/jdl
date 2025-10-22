package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.linear.LinearRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.linear.optimizer.ClosedSolutionOptimizer;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.plots.Plot;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ModelComplexity {

    public static void main(String[] args) {
        showForNumberType(JavaNumberTypeSupport.RATIONAL_BIGINT);
    }

    public static <N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport) {
        LinearRegression<N> regression = createClosedSolutionFitting(typeSupport);

        String titlePostfix = " (" + typeSupport.name() + ")";
        Plot errorPlot = new Plot("RME per N" + titlePostfix);
        Plot weightPlot = new Plot("Absolute sum of weights" + titlePostfix);
        Plot fitPlot = new Plot("Estimation per N" + titlePostfix);

        fitPlot.setYRange(-2, 2);
        fitPlot.plotTargetOutput(SyntheticDataSets.sinus(typeSupport).getEstimationFunction(), "Target");
        fitPlot.plotDataSet(regression.getTrainSet(), "Train Set");

        for (int m = 0; m < 10; m++) {
            LinearBasisFunctionModel<N> model = regression.fitComplexity(m);
            if (m == 0 || m % 2 != 0) {
                fitPlot.plotTargetOutput(model, "Complexity " + m);
            }

            float weights = 0.0f;
            List<Float> floats = new ArrayList<>();
            for (N weight : model.getParameters()) {
                weights += model.getNumberTypeSupport().absolute(weight).floatValue();
                floats.add(weight.floatValue());
            }
            System.out.println("m = " + m + "; weights: " + floats);
            weightPlot.addToSeries("Total Weights", new float[]{m}, new float[]{weights});

            Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
            errorPlot.addToSeries("Train RME", new float[]{m}, new float[]{currentRmes.left()});
            for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                errorPlot.addToSeries(currentRmeEntry.getKey() + " RME", new float[]{m}, new float[]{currentRmeEntry.getValue()});
            }
        }

        errorPlot.display();
        weightPlot.display();
        fitPlot.display();
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
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, new ClosedSolutionOptimizer<>(basisFunctionGenerator.getNumberTypeSupport()), Regimes.oneShot());
    }

}
