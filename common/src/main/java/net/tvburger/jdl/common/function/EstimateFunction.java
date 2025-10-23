package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.MarkerInterface;

@MarkerInterface
public interface EstimateFunction<TI extends Tensor<I>, I, TO extends Tensor<O>, O> extends TensorFunction<TI, I, TO, O> {
}
