package net.tvburger.jdl.model;

import net.tvburger.jdl.common.function.ScalarFunction;
import net.tvburger.jdl.common.numbers.Scalar;
import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.MarkerInterface;

@MarkerInterface
public interface ParameterizedTrainableScalarEstimationFunction<N extends Number, I extends Tensor<N>> extends ParameterizedTrainableEstimationFunction<N, I, Scalar<N>>, ScalarFunction<I, N, N> {
}
