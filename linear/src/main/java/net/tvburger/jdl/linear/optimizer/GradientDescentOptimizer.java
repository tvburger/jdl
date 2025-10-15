package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public interface GradientDescentOptimizer<N extends Number> extends LinearModelOptimizer<N>, IterativeOptimizer<N> {

    static <N extends Number> Interceptor<N> nullInterceptor() {
        return (epoch, model, trainSet, optimizer, gradients) -> gradients;
    }

    @FunctionalInterface
    interface Interceptor<N extends Number> {

        @SafeVarargs
        static <N extends Number> Interceptor<N> of(Interceptor<N>... interceptor) {
            return (epoch, model, trainSet, optimizer, gradients) -> {
                Vector<N> currentGradients = Vectors.convert(gradients, model.getCurrentNumberType());
                for (Interceptor<N> inter : interceptor) {
                    Vector<N> newGradients = inter.interceptGradients(epoch, model, trainSet, optimizer, gradients);
                    Vector<N> substract = newGradients.substract(gradients);
                    currentGradients = currentGradients.add(substract);
                }
                return currentGradients;
            };
        }

        Vector<N> interceptGradients(int epoch, LinearBasisFunctionModel<N> model, DataSet<N> trainSet, GradientDescentOptimizer<N> optimizer, Vector<N> gradients);

    }

    void setInterceptor(Interceptor<N> interceptor);

    default void setInterceptors(Interceptor<N>... interceptors) {
        setInterceptor(Interceptor.of(interceptors));
    }

    Interceptor<N> getInterceptor();

}
