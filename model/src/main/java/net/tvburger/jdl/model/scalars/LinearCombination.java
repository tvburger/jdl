package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * A simple linear scalar model of the form:
 * <pre>
 *   f(x) = bias + Σ w_i * x_i
 * </pre>
 * where the bias and weights are trainable parameters.
 * <p>
 * This class supports forward estimation of the scalar output as well as
 * calculation of parameter gradients required for gradient-based training.
 */
@Strategy(Strategy.Role.CONCRETE)
public class LinearCombination extends LinearModel implements TrainableScalarFunction {

    /**
     * Creates a new {@code LinearCombination} with the specified number of input dimensions.
     * <p>
     * The underlying parameter vector will have length {@code dimensions + 1},
     * with the first element reserved for the bias and the rest for the weights.
     *
     * @param dimensions number of input features
     * @return a new {@code LinearCombination} instance with zero-initialized parameters
     */
    public static LinearCombination create(int dimensions) {
        return new LinearCombination(new float[dimensions + 1]);
    }

    /**
     * Constructs a {@code LinearCombination} with the given parameter vector.
     * <p>
     * The first element of the array is treated as the bias, while the subsequent
     * elements are the weights.
     *
     * @param parameters parameter vector (bias + weights)
     */
    public LinearCombination(float[] parameters) {
        super(parameters);
    }

    /**
     * Estimates the scalar output of this linear model for the given input vector.
     * <p>
     * The computation is:
     * <pre>
     *   output = bias + Σ (inputs[i] * weights[i])
     * </pre>
     *
     * @param inputs the input feature vector
     * @return the scalar output value
     * @throws IllegalArgumentException if {@code inputs.length != arity()}
     */
    @Override
    public float estimateScalar(float[] inputs) {
        if (inputs.length != arity()) {
            throw new IllegalArgumentException();
        }
        float sum = getBias();
        for (int d = 1; d <= arity(); d++) {
            sum += inputs[d - 1] * getWeight(d);
        }
        return sum;
    }

    /**
     * Calculates the gradients of the output with respect to all parameters (bias and weights)
     * for the given input vector.
     * <p>
     * The gradient vector has length {@code 1 + arity()}, where:
     * <ul>
     *   <li>Index 0 corresponds to the derivative w.r.t. the bias (always {@code 1}).</li>
     *   <li>Indices 1..arity correspond to the input values, i.e. ∂f/∂w_i = input[i-1].</li>
     * </ul>
     *
     * @param inputs the input feature vector
     * @return the gradient vector (bias gradient followed by weight gradients)
     */
    public float[] calculateParameterGradients(float[] inputs) {
        float[] gradients = new float[1 + arity()];
        gradients[0] = 1;
        if (arity() >= 0) System.arraycopy(inputs, 0, gradients, 1, arity());
        return gradients;
    }
}
