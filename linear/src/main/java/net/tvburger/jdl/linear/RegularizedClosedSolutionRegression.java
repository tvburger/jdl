package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linalg.*;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.DataSet;

import java.util.Map;

public class RegularizedClosedSolutionRegression<N extends Number> extends ClosedSolutionRegression<N> {

    private float lambda;

    public RegularizedClosedSolutionRegression(SyntheticDataSets.SyntheticDataSet<N> targetFit, BasisFunction.Generator<N> basisFunctionGenerator, DataSet<N> trainSet, Map<String, DataSet<N>> testSets) {
        super(targetFit, basisFunctionGenerator, trainSet, testSets);
    }

    public void setLambda(float lambda) {
        this.lambda = lambda;
    }

    public float getLambda() {
        return lambda;
    }

    public void fitComplexity(int m) {
        if (lambda == 0.0f) {
            lambda = 1e-15f;
        }
        super.fitComplexity(m);
        LinearBasisFunctionModel<N> regression = fitted.remove(m);
        int key = (int) Math.log(lambda);
        System.out.println("Index : " + key);
        fitted.put(key, regression);
    }

    @Override
    protected void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet) {
//        if (lambda == 0) {
//            super.setOptimalWeights(regression, trainSet);
//            return;
//        }
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
                        .multiply(typeSupport.valueOf(lambda)))
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

}
