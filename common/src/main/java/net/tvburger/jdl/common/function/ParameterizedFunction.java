package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.Strategy;

@Strategy(Strategy.Role.INTERFACE)
public interface ParameterizedFunction<P, TI extends Tensor<I>, I, TO extends Tensor<O>, O> extends TensorFunction<TI, I, TO, O> {

    int getParameterCount();

    Array<P> getParameters();

    void setParameters(Array<P> parameters);

    default P getParameter(int i) {
        return getParameters().get(i);
    }

    default void setParameter(int i, P parameter) {
        getParameters().set(i, parameter);
    }

}
