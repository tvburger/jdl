package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.function.Function;

@Strategy(Strategy.Role.INTERFACE)
public interface TensorFunction<TI extends Tensor<I>, I, TO extends Tensor<O>, O> extends Function<TI, TO> {

    int arity();

    int coArity();

}
