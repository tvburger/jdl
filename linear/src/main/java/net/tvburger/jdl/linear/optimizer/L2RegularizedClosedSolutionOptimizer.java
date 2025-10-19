package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.*;
import net.tvburger.jdl.linear.FeatureMatrices;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.optimizer.UpdateStep;

public class L2RegularizedClosedSolutionOptimizer<N extends Number> implements LinearModelOptimizer<N> {

    private boolean debugOutput;
    private N lambda;

    private final JavaNumberTypeSupport<N> typeSupport;

    public L2RegularizedClosedSolutionOptimizer(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return typeSupport;
    }

    public N getLambda() {
        return lambda;
    }

    public void setLambda(N lambda) {
        this.lambda = lambda;
    }

    public boolean isDebugOutput() {
        return debugOutput;
    }

    public void setDebugOutput(boolean debugOutput) {
        this.debugOutput = debugOutput;
    }

    @Override
    public void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainingSet) {
        JavaNumberTypeSupport<N> typeSupport = regression.getCurrentNumberType();
        if (debugOutput) {
            System.out.println("Number type = " + typeSupport.name());
        }

        N[] values = typeSupport.createArray(trainingSet.size());
        for (int i = 0; i < values.length; i++) {
            values[i] = trainingSet.samples().get(i).targetOutputs()[0];
        }
        TypedVector<N> y = Vectors.of(typeSupport, values).transpose();
        if (debugOutput) {
            y.print("y");
        }
        Matrix<N> designMatrix = FeatureMatrices.create(regression.getFeatureExtractor(), trainingSet);
        if (debugOutput) {
            designMatrix.print("Φ");
        }

        Matrix<N> transposedDesignMatrix = designMatrix.transpose();

        Matrix<N> regularizedInvertedDesignMatrix = transposedDesignMatrix
                .multiply(designMatrix)
                .add(Matrices.identity(designMatrix.m(), typeSupport)
                        .multiply(lambda))
                .invert()
                .multiply(transposedDesignMatrix);
        if (debugOutput) {
            regularizedInvertedDesignMatrix.print("(Φ" + Notations.TRANSPOSED + "Φ + " + Notations.LAMBDA + "I)" + Notations.INVERSE);
        }

        Vector<N> weights = regularizedInvertedDesignMatrix.multiply(y);
        if (debugOutput) {
            weights.print("w");
        }

        for (int i = 0; i < weights.getDimensions(); i++) {
            regression.setParameter(i, weights.get(i + 1));
        }
    }

}
