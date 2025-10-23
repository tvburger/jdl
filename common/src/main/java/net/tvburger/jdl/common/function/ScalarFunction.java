package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Scalar;
import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.Strategy;

@Strategy(Strategy.Role.INTERFACE)
public interface ScalarFunction<TI extends Tensor<I>, I, O> extends TensorFunction<TI, I, Scalar<O>, O> {

    default int coArity() {
        return 1;
    }

}
