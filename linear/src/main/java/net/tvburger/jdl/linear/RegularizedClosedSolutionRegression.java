package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.linalg.*;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.DataSet;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

public class RegularizedClosedSolutionRegression<N extends Number> {

    public boolean DEBUG_OUTPUT = true;

    private final Map<Double, LinearBasisFunctionModel<N>> fitted = new TreeMap<>();

    protected final BasisFunction.Generator<N> basisFunctionGenerator;
    private final DataSet<N> trainSet;
    private final Map<String, DataSet<N>> testSets;


    public RegularizedClosedSolutionRegression(BasisFunction.Generator<N> basisFunctionGenerator, DataSet<N> trainSet, Map<String, DataSet<N>> testSets) {
        this.basisFunctionGenerator = basisFunctionGenerator;
        this.trainSet = trainSet;
        this.testSets = testSets;
    }

    public LinearBasisFunctionModel<N> getUnregularizedModel(int m) {
        LinearBasisFunctionModel<N> regression = LinearBasisFunctionModel.create(m, basisFunctionGenerator);
        setOptimalWeights(regression, trainSet, basisFunctionGenerator.getCurrentNumberType().zero());
        return regression;
    }

    public LinearBasisFunctionModel<N> fitComplexity(int m, N lambda) {
        if (!basisFunctionGenerator.getCurrentNumberType().greaterThan(lambda, basisFunctionGenerator.getCurrentNumberType().zero())) {
            throw new IllegalArgumentException("Lambda must be > 0!");
        }
        LinearBasisFunctionModel<N> regression = LinearBasisFunctionModel.create(m, basisFunctionGenerator);
        setOptimalWeights(regression, trainSet, lambda);
        double key = Math.log10(lambda.doubleValue());
        fitted.put(key, regression);
        return regression;
    }

    private void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet, N lambda) {
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

        Matrix<N> transposedDesignMatrix = designMatrix.transpose();

        Matrix<N> regularizedInvertedDesignMatrix = transposedDesignMatrix
                .multiply(designMatrix)
                .add(Matrices.identity(designMatrix.m(), typeSupport)
                        .multiply(lambda))
                .invert()
                .multiply(transposedDesignMatrix);
        if (DEBUG_OUTPUT) {
            regularizedInvertedDesignMatrix.print("(Φ" + Notations.TRANSPOSED + "Φ + " + Notations.LAMBDA + "I)" + Notations.INVERSE);
        }

        Vector<N> weights = regularizedInvertedDesignMatrix.multiply(y);
        if (DEBUG_OUTPUT) {
            weights.print("w");
        }

        for (int i = 0; i < weights.getDimensions(); i++) {
            regression.setParameter(i, weights.get(i + 1));
        }
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
}
