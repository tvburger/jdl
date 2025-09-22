package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.ClosedSolutionRegression;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.scalars.UnaryEstimationFunction;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class TrainingSetSize {

    public static <N extends Number> void main(String[] args) {
        List<JavaNumberTypeSupport<?>> types = List.of(JavaNumberTypeSupport.DOUBLE, JavaNumberTypeSupport.RATIONAL_BIGINT);

        Plot fitPlot = new Plot("Fit vs Training Set Size");
        UnaryEstimationFunction<Double> targetFit = SyntheticDataSets.sinus(JavaNumberTypeSupport.DOUBLE).getEstimationFunction();
        fitPlot.plotTargetOutput(targetFit, "Target");
        fitPlot.display();

        Plot mrePlot = new Plot("RME");
        mrePlot.display();

        for (int trainingSetSize = 10; trainingSetSize <= 1000; trainingSetSize += trainingSetSize > 100 ? 100 : 10) {
            for (JavaNumberTypeSupport<?> type : types) {
                ClosedSolutionRegression<N> regression = getFittedFunction(type, trainingSetSize);
                LinearBasisFunctionModel<N> model = regression.getModel(9);
                fitPlot.plotTargetOutput(model, "Fit " + type.name());
                Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
                mrePlot.addToSeries(type.name() + " Train RME", new float[]{trainingSetSize}, new float[]{currentRmes.left()});
                for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                    mrePlot.addToSeries(type.name() + " " + currentRmeEntry.getKey() + " RME", new float[]{trainingSetSize}, new float[]{currentRmeEntry.getValue()});
                }
            }
            fitPlot.getChart().setTitle("Training Set Size = " + trainingSetSize);
            fitPlot.redraw();
            mrePlot.redraw();
        }
    }

    @SuppressWarnings("unchecked")
    private static <N extends Number> ClosedSolutionRegression<N> getFittedFunction(JavaNumberTypeSupport<?> typeSupport, int trainingSetSize) {
        JavaNumberTypeSupport<N> typedTypeSupport = (JavaNumberTypeSupport<N>) typeSupport;
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typedTypeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typedTypeSupport);
        return create(dataSetGenerator, basisFunctionGenerator, trainingSetSize);
    }

    public static <N extends Number> ClosedSolutionRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator, int trainingSetSize) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(0.1f);
        DataSet<N> trainSet = dataSetGenerator.load(trainingSetSize);
        dataSetGenerator.setNoiseScale(0.1f);
        DataSet<N> testSet = dataSetGenerator.load(1000);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new ClosedSolutionRegression<>(dataSetGenerator, basisFunctionGenerator, trainSet, testSets);
    }

}
