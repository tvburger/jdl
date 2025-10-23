package net.tvburger.jdl.model;

import net.tvburger.jdl.common.function.EstimateFunction;
import net.tvburger.jdl.common.function.ParameterizedTrainableFunction;
import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.MarkerInterface;

@MarkerInterface
public interface ParameterizedTrainableEstimationFunction<N extends Number, I extends Tensor<N>, O extends Tensor<N>> extends ParameterizedTrainableFunction<N, I, N, O, N>, EstimateFunction<I, N, O, N>, NumberTypeAgnostic<N> {
}
