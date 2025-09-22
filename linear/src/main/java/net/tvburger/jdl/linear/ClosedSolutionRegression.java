package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linalg.*;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.plots.Plot;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

public class ClosedSolutionRegression<N extends Number> {

    public boolean DEBUG_OUTPUT = false;

    private final SyntheticDataSets.SyntheticDataSet<N> targetFit;
    private final BasisFunction.Generator<N> basisFunctionGenerator;
    private final DataSet<N> trainSet;
    private final Map<String, DataSet<N>> testSets;

    protected final Map<Integer, LinearBasisFunctionModel<N>> fitted = new TreeMap<>();

    private final Plot errorPlot;
    private final Plot fitPlot;
    private final Plot weightPlot;

    private boolean showTestSets;
    private boolean showTrainSet;
    private boolean showTargetFit;

    public ClosedSolutionRegression(SyntheticDataSets.SyntheticDataSet<N> targetFit, BasisFunction.Generator<N> basisFunctionGenerator, DataSet<N> trainSet, Map<String, DataSet<N>> testSets) {
        this.basisFunctionGenerator = basisFunctionGenerator;
        this.targetFit = targetFit;
        this.trainSet = trainSet;
        this.testSets = testSets;

        String titlePostfix = " (" + basisFunctionGenerator.getCurrentNumberType().name() + ")";
        errorPlot = new Plot("RME per N" + titlePostfix);
        fitPlot = new Plot("Estimation per N" + titlePostfix);
        weightPlot = new Plot("Absolute sum of weights" + titlePostfix);
    }

    public LinearBasisFunctionModel<N> getModel(int m) {
        if (!fitted.containsKey(m)) {
            fitComplexity(m);
        }
        return fitted.get(m);
    }

    public void fitComplexity(int m) {
        LinearBasisFunctionModel<N> regression = LinearBasisFunctionModel.create(m, basisFunctionGenerator);
        setOptimalWeights(regression, trainSet);
        fitted.put(m, regression);
    }

    protected void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet) {
        JavaNumberTypeSupport<N> typeSupport = regression.getCurrentNumberType();
        if (DEBUG_OUTPUT) {
            System.out.println("Number type = " + typeSupport.name());
        }

        N[] values = typeSupport.createArray(trainSet.size());
        for (int i = 0; i < values.length; i++) {
            values[i] = trainSet.samples().get(i).targetOutputs()[0];
        }
        TypedVector<N> y = Vectors.of(typeSupport, values).transpose();
        if (DEBUG_OUTPUT) {
            y.print("y");
        }
        Matrix<N> designMatrix = FeatureMatrices.create(regression.getFeatureExtractor(), trainSet);
        if (DEBUG_OUTPUT) {
            designMatrix.print("Φ");
        }

        Matrix<N> invertedDesignMatrix = designMatrix.pseudoInvert();
        if (DEBUG_OUTPUT) {
            invertedDesignMatrix.multiply(designMatrix).print("Φ" + Notations.PSEUDO_INVERSE + "Φ = I");
        }

        Vector<N> weights = invertedDesignMatrix.multiply(y);
        if (DEBUG_OUTPUT) {
            weights.print("w");
        }

        for (int i = 0; i < weights.getDimensions(); i++) {
            regression.setParameter(i, weights.get(i + 1));
        }
    }

    public void showErrorPlot() {
        float[] x = new float[fitted.size()];
        float[] trainRme = new float[x.length];
        Map<String, float[]> testRmes = new LinkedHashMap<>(testSets.size());
        for (Map.Entry<String, DataSet<N>> testSetEntry : testSets.entrySet()) {
            testRmes.put(testSetEntry.getKey(), new float[x.length]);
        }
        int i = 0;
        for (Map.Entry<Integer, LinearBasisFunctionModel<N>> fittedEntry : fitted.entrySet()) {
            x[i] = (float) fittedEntry.getKey();
            trainRme[i] = calculateRME(trainSet, fittedEntry.getValue());
            for (Map.Entry<String, DataSet<N>> testSetEntry : testSets.entrySet()) {
                testRmes.get(testSetEntry.getKey())[i] = calculateRME(testSetEntry.getValue(), fittedEntry.getValue());
            }
            i++;
        }
        errorPlot.setSeries("Train Set", x, trainRme);
        testRmes.forEach((k, v) -> errorPlot.setSeries(k, x, v));
        errorPlot.display();
    }

    public Pair<Float, Map<String, Float>> calculateRMEs(LinearBasisFunctionModel<N> model) {
        float trainRme = calculateRME(trainSet, model);
        Map<String, Float> testRmes = new LinkedHashMap<>();
        for (Map.Entry<String, DataSet<N>> testSetEntry : testSets.entrySet()) {
            testRmes.put(testSetEntry.getKey(), calculateRME(testSetEntry.getValue(), model));
        }
        return Pair.of(trainRme, testRmes);
    }

    private float calculateRME(DataSet<N> dataSet, LinearBasisFunctionModel<N> regression) {
        float mse = 0.0f;
        for (DataSet.Sample<N> sample : dataSet) {
            float estimated = regression.estimateScalar(sample.features()).floatValue();
            float target = sample.targetOutputs()[0].floatValue();
            mse += (float) Math.pow(estimated - target, 2) / dataSet.size();
        }
        return (float) Math.sqrt(mse);
    }

    public void showFitPlot() {
        showFitPlot(1000);
    }

    public void showFitPlot(int n) {
        if (showTestSets) {
            for (Map.Entry<String, DataSet<N>> testSetEntry : testSets.entrySet()) {
                fitPlot.plotDataSet(testSetEntry.getValue(), testSetEntry.getKey());
            }
        }
        for (Map.Entry<Integer, LinearBasisFunctionModel<N>> fittedEntry : fitted.entrySet()) {
            fitPlot.plotTargetOutput(fittedEntry.getValue(), "Complexity " + fittedEntry.getKey(), n);
        }
        if (showTargetFit) {
            fitPlot.plotTargetOutput(targetFit.getEstimationFunction(), "Target Fit", n);
        }
        if (showTrainSet) {
            fitPlot.plotDataSet(trainSet, "Train Set");
        }
        fitPlot.display();
    }

    public void showWeightsPlot() {
        JavaNumberTypeSupport<N> supportType = basisFunctionGenerator.getCurrentNumberType();
        float[] x = new float[fitted.size()];
        float[] y = new float[x.length];
        int i = 0;
        for (Map.Entry<Integer, LinearBasisFunctionModel<N>> weightsEntry : fitted.entrySet()) {
            x[i] = weightsEntry.getKey();
            N[] parameters = weightsEntry.getValue().getParameters();
            y[i] = 0.0f;
            for (int p = 0; p < parameters.length; p++) {
                y[i] += supportType.absolute(parameters[p]).floatValue();
            }
            i++;
        }
        weightPlot.setSeries("Absolute Weights", x, y);
        weightPlot.display();
    }

    public boolean isShowTestSets() {
        return showTestSets;
    }

    public void setShowTestSets(boolean showTestSets) {
        this.showTestSets = showTestSets;
    }

    public boolean isShowTrainSet() {
        return showTrainSet;
    }

    public void setShowTrainSet(boolean showTrainSet) {
        this.showTrainSet = showTrainSet;
    }

    public boolean isShowTargetFit() {
        return showTargetFit;
    }

    public void setShowTargetFit(boolean showTargetFit) {
        this.showTargetFit = showTargetFit;
    }

}
