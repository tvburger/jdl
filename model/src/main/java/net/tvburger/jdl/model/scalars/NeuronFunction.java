package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.scalars.activations.ActivationFunction;

/**
 * A neuron-like scalar function that combines a {@link LinearCombination}
 * with a non-linear {@link ActivationFunction}.
 * <p>
 * The computation is:
 * <pre>
 *   output = activation( bias + Σ w_i * x_i )
 * </pre>
 * where the activation function is applied to the linear combination of inputs.
 * <p>
 * Provides parameter gradient calculations that include the effect of the
 * activation function, making this class suitable for use in neural network
 * training via backpropagation.
 */
@Strategy(Strategy.Role.CONCRETE)
public class NeuronFunction extends LinearCombination {

    private final ActivationFunction activationFunction;

    /**
     * Creates a new neuron function with the given parameters and activation.
     * <p>
     * The parameter array follows the {@link LinearCombination} convention:
     * index {@code 0} is the bias, indices {@code 1..} are the weights.
     *
     * @param parameters         parameter vector (bias + weights)
     * @param activationFunction the non-linear activation function to apply
     */
    public NeuronFunction(float[] parameters, ActivationFunction activationFunction) {
        super(parameters);
        this.activationFunction = activationFunction;
    }

    /**
     * Returns the activation function applied to the weighted sum.
     *
     * @return the activation function
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * Estimates the scalar output by first computing the weighted sum
     * (via {@link LinearCombination#estimateScalar(float[])})
     * and then applying the activation function.
     *
     * @param inputs the input feature vector
     * @return the activated scalar output
     * @throws IllegalArgumentException if {@code inputs.length != arity()}
     */
    @Override
    public float estimateScalar(float[] inputs) {
        return activationFunction.activate(super.estimateScalar(inputs));
    }

    /**
     * Calculates the gradients of the output with respect to all parameters,
     * including the effect of the activation function.
     * <p>
     * Concretely:
     * <ul>
     *   <li>First calls {@link #calculateParameterGradients_df_dp(float[])}
     *       to compute gradients of the pre-activation linear function
     *       (bias + weighted inputs).</li>
     *   <li>Then multiplies each gradient by the derivative of the activation
     *       function evaluated at the current output.</li>
     * </ul>
     *
     * @param inputs the input feature vector
     * @return the gradient vector (bias gradient followed by weight gradients)
     */
    public float[] calculateParameterGradients(float[] inputs) {
        float[] gradients = calculateParameterGradients_df_dp(inputs);
        float activationGradient = activationFunction.determineGradientForOutput(estimateScalar(inputs));
        for (int p = 0; p < gradients.length; p++) {
            gradients[p] *= activationGradient;
        }
        return gradients;
    }

    /**
     * Calculates the parameter gradients of the underlying linear combination
     * (pre-activation function). This is equivalent to
     * {@link LinearCombination#calculateParameterGradients(float[])}.
     *
     * @param inputs the input feature vector
     * @return the gradient vector of the linear function
     */
    public float[] calculateParameterGradients_df_dp(float[] inputs) {
        return super.calculateParameterGradients(inputs);
    }

}
