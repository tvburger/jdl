package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.*;
import net.tvburger.jdl.linear.FeatureMatrices;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public class L2Regularization<N extends Number> implements GradientDescentOptimizer.Interceptor<N> {

    private N lambda;

    public L2Regularization(JavaNumberTypeSupport<N> typeSupport) {
        this(typeSupport.zero());
    }

    public L2Regularization(N lambda) {
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
        N[] parameters = model.getParameters();
        Vector<N> thetas = Vectors.of(model.getCurrentNumberType(), parameters).transpose();
        Vector<N> regularization = thetas.multiply(lambda);
        return gradients.add(regularization);
    }
}