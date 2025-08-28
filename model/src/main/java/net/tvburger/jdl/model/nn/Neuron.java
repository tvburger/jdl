package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.model.nn.activations.ActivationFunction;

import java.util.Arrays;
import java.util.List;

/**
 * Represents a single artificial neuron in a neural network.
 * <p>
 * A {@code Neuron} receives inputs from other neurons, combines them
 * using a weighted sum plus a bias, and passes the result (the "logit")
 * through an {@link ActivationFunction} to produce its output.
 * </p>
 *
 * <h2>Computation</h2>
 * For a neuron with inputs {@code x₁ … xₙ}, weights {@code w₁ … wₙ},
 * and bias {@code b}, the forward activation is:
 * <pre>
 *   logit = b + Σ (wᵢ * xᵢ)
 *   output = activationFunction.activate(logit)
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * <ul>
 *   <li>Before activation, the neuron has no valid {@code logit} or {@code output}.</li>
 *   <li>{@link #activate()} computes the neuron’s output, caching
 *       the logit and activation result and marking it as "activated".</li>
 *   <li>{@link #deactivate()} resets the activation flag, allowing the neuron
 *       to be re-evaluated (e.g., for another forward pass).</li>
 *   <li>{@link #getLogit()} and {@link #getOutput()} may only be called
 *       after activation.</li>
 * </ul>
 *
 * <h2>Thread safety</h2>
 * <ul>
 *   <li>Activation and deactivation are synchronized to prevent race conditions
 *       in concurrent evaluation.</li>
 *   <li>Weight and bias setters are not synchronized; concurrent modification
 *       must be managed externally if required.</li>
 * </ul>
 *
 * <h2>Typical usage:</h2>
 * <pre>{@code
 * Neuron n1 = new Neuron("input1", List.of(), new Identity());
 * Neuron n2 = new Neuron("input2", List.of(), new Identity());
 * Neuron hidden = new Neuron("hidden", List.of(n1, n2), new Sigmoid());
 *
 * // set inputs manually
 * n1.setBias(1.0f); n1.activate(); // identity, so output = bias
 * n2.setBias(2.0f); n2.activate();
 *
 * hidden.setWeights(new float[]{0.5f, -0.3f});
 * hidden.setBias(0.1f);
 * hidden.activate();
 *
 * System.out.println(hidden.getOutput());
 * }</pre>
 */
@DomainObject
public class Neuron {

    private final String name;
    private final List<? extends Neuron> inputs;
    private float[] weights;
    private float bias;
    /**
     * Linear combination of inputs before activation.
     */
    protected float logit;
    private final ActivationFunction activationFunction;
    /**
     * Cached output after applying the activation function.
     */
    protected float output;
    /**
     * Flag indicating whether {@link #activate()} has been called.
     */
    protected boolean activated;

    /**
     * Constructs a new neuron with the given name, inputs, and activation function.
     *
     * @param name               human-readable identifier
     * @param inputs             input neurons (empty if none)
     * @param activationFunction function to apply to the weighted sum
     */
    public Neuron(String name, List<? extends Neuron> inputs, ActivationFunction activationFunction) {
        this.name = name;
        this.inputs = inputs == null ? List.of() : inputs;
        this.weights = new float[this.inputs.size()];
        this.activationFunction = activationFunction;
    }

    /**
     * @return the human-readable identifier of this neuron
     */
    public String getName() {
        return name;
    }

    /**
     * @return the input neurons feeding into this neuron
     */
    public List<? extends Neuron> getInputs() {
        return inputs;
    }

    /**
     * @return the activation function applied to the weighted sum
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * Computes the logit and output of this neuron by aggregating inputs,
     * applying weights and bias, and passing the result through the
     * activation function. Subsequent calls are ignored until
     * {@link #deactivate()} is invoked.
     */
    public synchronized void activate() {
        if (isActivated()) {
            return;
        }
        logit = bias;
        for (int i = 0; i < inputs.size(); i++) {
            float input = inputs.get(i).getOutput();
            logit += input * weights[i];
        }
        output = activationFunction.activate(logit);
        activated = true;
    }

    /**
     * @return whether this neuron has been activated
     */
    public boolean isActivated() {
        return activated;
    }

    /**
     * Resets the activated flag so the neuron can be re-evaluated.
     */
    public synchronized void deactivate() {
        activated = false;
    }

    /**
     * @return the trainable weights of this neuron
     */
    public float[] getWeights() {
        return weights;
    }

    /**
     * Sets the trainable weights of this neuron.
     */
    public void setWeights(float[] weights) {
        this.weights = weights;
    }

    /**
     * @return the trainable bias of this neuron
     */
    public float getBias() {
        return bias;
    }

    /**
     * Sets the trainable bias of this neuron.
     */
    public void setBias(float bias) {
        this.bias = bias;
    }

    /**
     * Returns the pre-activation weighted sum of inputs.
     *
     * @return the computed logit
     * @throws IllegalStateException if the neuron has not been activated
     */
    public float getLogit() {
        if (!isActivated()) {
            throw new IllegalStateException("Neuron not activated!");
        }
        return logit;
    }

    /**
     * Returns the post-activation output of this neuron.
     *
     * @return the activated output value
     * @throws IllegalStateException if the neuron has not been activated
     */
    public float getOutput() {
        if (!isActivated()) {
            throw new IllegalStateException("Neuron not activated!");
        }
        return output;
    }

    /**
     * Finds the weight associated with a given source neuron.
     *
     * @param source the input neuron
     * @return the corresponding weight, or {@code null} if not found
     */
    public Float findWeight(Neuron source) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i) == source) {
                return weights[i];
            }
        }
        return null;
    }

    /**
     * @return string representation including name, activation state, logit, output, weights, and bias
     */
    @Override
    public String toString() {
        return name + "{" + activated + ", " + logit + ", " + output + "}" + Arrays.toString(weights) + (bias >= 0.0 ? "+" : "") + bias;
    }

}
