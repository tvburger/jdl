package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public class ElasticNet<N extends Number> implements GradientDescentOptimizer.Interceptor<N> {

    private final L1Regularization<N> l1;
    private final L2Regularization<N> l2;
    private final GradientDescentOptimizer.Interceptor<N> interceptor;

    public ElasticNet(JavaNumberTypeSupport<N> numberTypeSupport) {
        this(numberTypeSupport.zero(), numberTypeSupport.zero());
    }

    public ElasticNet(N lambda1, N lambda2) {
        this.l1 = new L1Regularization<>(lambda1);
        this.l2 = new L2Regularization<>(lambda2);
        this.interceptor = GradientDescentOptimizer.Interceptor.of(l1, l2);
    }

    public void setLambda1(N lambda1) {
        this.l1.setLambda(lambda1);
    }

    public N getLambda1() {
        return this.l1.getLambda();
    }

    public void setLambda2(N lambda2) {
        this.l2.setLambda(lambda2);
    }

    public N getLambda2() {
        return this.l2.getLambda();
    }

    @Override
    public Vector<N> interceptGradients(int epoch, LinearBasisFunctionModel<N> model, DataSet<N> trainSet, GradientDescentOptimizer<N> optimizer, Vector<N> gradients) {
        return interceptor.interceptGradients(epoch, model, trainSet, optimizer, gradients);
    }
}
