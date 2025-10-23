package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.model.ParameterizedTrainableEstimationFunction;

public interface NeuralNetworkLayer<N extends Number, I extends Tensor<N>, O extends Tensor<N>> extends ParameterizedTrainableEstimationFunction<N, I, O> {
}
