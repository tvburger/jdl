package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Tensor;

public interface ParameterizedTrainableFunction<P, TI extends Tensor<I>, I, TO extends Tensor<O>, O> extends ParameterizedFunction<P, TI, I, TO, O>, TrainableFunction<TI, I, TO, O>, DifferentiableFunction<TI, I, TO, O> {

    Tensor<P> getParameterGradients(Tensor<O> errorSignals);

}
