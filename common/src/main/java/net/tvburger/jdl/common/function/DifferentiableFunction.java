package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Tensor;

public interface DifferentiableFunction<TI extends Tensor<I>, I, TO extends Tensor<O>, O> extends TensorFunction<TI, I, TO, O> {

    Tensor<I> getInputGradients(Tensor<O> errorSignals);

}
