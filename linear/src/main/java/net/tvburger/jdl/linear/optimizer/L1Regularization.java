package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.Matrices;
import net.tvburger.jdl.linalg.TypedMatrix;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public class L1Regularization<N extends Number> implements GradientDescentOptimizer.Interceptor<N> {

    private N lambda;

    public L1Regularization(JavaNumberTypeSupport<N> typeSupport) {
        this(typeSupport.zero());
    }

    public L1Regularization(N lambda) {
        this.lambda = lambda;
    }

    public N getLambda() {
        return lambda;
    }

    public void setLambda(N lambda) {
        this.lambda = lambda;
    }

    @Override
    public Vector<N> interceptGradients(int epoch, LinearBasisFunctionModel<N> model, DataSet<N> trainSet, GradientDescentOptimizer<N> optimizer, Vector<N> gradients) {
        JavaNumberTypeSupport<N> currentNumberType = model.getCurrentNumberType();
        N[] parameters = model.getParameters();
        N[] regularization = currentNumberType.createArray(parameters.length);
        N negativeLambda = currentNumberType.negate(lambda);
        for (int p = 0; p < parameters.length; p++) {
            regularization[p] = currentNumberType.isZero(parameters[p])
                    ? currentNumberType.zero()
                    : currentNumberType.positive(parameters[p])
                    ? lambda : negativeLambda;
        }
        return gradients.add(Vectors.of(currentNumberType, regularization).transpose());
    }
}