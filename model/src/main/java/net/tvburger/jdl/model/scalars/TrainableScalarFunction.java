package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.training.TrainableFunction;

/**
 * A scalar-valued function that is both {@link TrainableFunction} and
 * {@link ScalarEstimationFunction}.
 * <p>
 * Implementations represent functions {@code f(x; θ)} with trainable parameters θ
 * that produce a scalar output for a given input vector. This interface provides
 * a way to compute parameter parameterGradients needed for training via backpropagation.
 */
@Strategy(Strategy.Role.INTERFACE)
public interface TrainableScalarFunction<N extends Number> extends TrainableFunction<N>, ScalarEstimationFunction<N> {

    /**
     * Calculates the parameterGradients of the scalar function with respect to its
     * trainable parameters, given the input values.
     *
     * @param inputs the input feature vector to the function
     * @return an array of partial derivatives
     * {@code ∂f/∂θ_k} for each parameter θ_k
     */
    Array<N> calculateParameterGradients(Array<N> inputs);

}
