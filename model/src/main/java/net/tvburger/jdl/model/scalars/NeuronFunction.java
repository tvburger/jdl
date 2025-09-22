package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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
public class NeuronFunction implements TrainableScalarFunction<Float> {

    private final LinearCombination<Float> linearCombination;
    private final ActivationFunction activationFunction;

    /**
     * Creates a new neuron function with the given parameters and activation.
     * The basis function used in the linear combination is the basis function.
     * <p>
     * The parameter array follows the {@link LinearCombination} convention:
     * index {@code 0} is the bias, indices {@code 1..} are the weights.
     *
     * @param dimensions         number of input dimensions
     * @param activationFunction the non-linear activation function to apply
     */
    public static NeuronFunction create(int dimensions, ActivationFunction activationFunction) {
        return new NeuronFunction(LinearCombination.create(dimensions, JavaNumberTypeSupport.FLOAT), activationFunction);
    }

    /**
     * Creates a new neuron function with the given parameters and activation.
     * <p>
     * The parameter array follows the {@link LinearCombination} convention:
     * index {@code 0} is the bias, indices {@code 1..} are the weights.
     *
     * @param linearCombination  the linear combination to apply to the inputs
     * @param activationFunction the non-linear activation function to apply
     */
    public NeuronFunction(LinearCombination<Float> linearCombination, ActivationFunction activationFunction) {
        this.linearCombination = linearCombination;
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
     * (via {@link LinearCombination#estimateScalar(Number[])})
     * and then applying the activation function.
     *
     * @param inputs the input feature vector
     * @return the activated scalar output
     * @throws IllegalArgumentException if {@code inputs.length != arity()}
     */
    @Override
    public Float estimateScalar(Float[] inputs) {
        return activationFunction.activate(linearCombination.estimateScalar(inputs));
    }

    /**
     * Calculates the gradients of the output with respect to all parameters,
     * including the effect of the activation function.
     * <p>
     * Concretely:
     * <ul>
     *   <li>First calls {@link #calculateParameterGradients_df_dp(Float[])}
     *       to compute gradients of the pre-activation linear function
     *       (bias + weighted inputs).</li>
     *   <li>Then multiplies each gradient by the derivative of the activation
     *       function evaluated at the current output.</li>
     * </ul>
     *
     * @param inputs the input feature vector
     * @return the gradient vector (bias gradient followed by weight gradients)
     */
    public Float[] calculateParameterGradients(Float[] inputs) {
        Float[] gradients = calculateParameterGradients_df_dp(inputs);
        float activationGradient = activationFunction.determineGradientForOutput(estimateScalar(inputs));
        for (int p = 0; p < gradients.length; p++) {
            gradients[p] *= activationGradient;
        }
        return gradients;
    }

    /**
     * Calculates the parameter gradients of the underlying linear combination
     * (pre-activation function). This is equivalent to
     * {@link LinearCombination#calculateParameterGradients(Number[])}.
     *
     * @param inputs the input feature vector
     * @return the gradient vector of the linear function
     */
    public Float[] calculateParameterGradients_df_dp(Float[] inputs) {
        return linearCombination.calculateParameterGradients(inputs);
    }

    @Override
    public JavaNumberTypeSupport<Float> getCurrentNumberType() {
        return JavaNumberTypeSupport.FLOAT;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int arity() {
        return linearCombination.arity();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Float[] getParameters() {
        return linearCombination.getParameters();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        Float[] parameters = getParameters();
        StringBuilder sb = new StringBuilder();
        sb.append(activationFunction.getClass().getSimpleName());
        sb.append("(").append(parameters[0] < 0.0f ? "" : '+').append(parameters[0]);
        for (int i = 1; i < parameters.length && i < 7; i++) {
            sb.append(':').append(String.format("%.2f", parameters[i]));
        }
        if (parameters.length > 7) {
            sb.append(":...");
        }
        return sb.append(')').toString();
    }
}
