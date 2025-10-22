package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.Matrix;
import net.tvburger.jdl.linalg.Notations;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linear.FeatureMatrices;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public class ClosedSolutionOptimizer<N extends Number> implements LinearModelOptimizer<N> {

    private boolean debugOutput;

    private final JavaNumberTypeSupport<N> typeSupport;

    public ClosedSolutionOptimizer(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }

    public boolean isDebugOutput() {
        return debugOutput;
    }

    public void setDebugOutput(boolean debugOutput) {
        this.debugOutput = debugOutput;
    }

    @Override
    public void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet) {
        JavaNumberTypeSupport<N> typeSupport = regression.getNumberTypeSupport();
        if (debugOutput) {
            System.out.println("Number type = " + typeSupport.name());
        }

        Array<N> values = typeSupport.createArray(trainSet.size());
        for (int i = 0; i < values.length(); i++) {
            values.set(i, trainSet.samples().get(i).targetOutputs().get(0));
        }
        TypedVector<N> y = new TypedVector<>(values, true, typeSupport);
        if (debugOutput) {
            y.print("y");
        }
        Matrix<N> designMatrix = FeatureMatrices.create(regression.getFeatureExtractor(), trainSet);
        if (debugOutput) {
            designMatrix.print("Φ");
        }

        Matrix<N> invertedDesignMatrix = designMatrix.pseudoInvert();
        if (debugOutput) {
            invertedDesignMatrix.multiply(designMatrix).print("Φ" + Notations.PSEUDO_INVERSE + "Φ = I");
        }

        Vector<N> weights = invertedDesignMatrix.multiply(y);
        if (debugOutput) {
            weights.print("w");
        }

        for (int i = 0; i < weights.getDimensions(); i++) {
            regression.setParameter(i, weights.get(i + 1));
        }
    }

}
