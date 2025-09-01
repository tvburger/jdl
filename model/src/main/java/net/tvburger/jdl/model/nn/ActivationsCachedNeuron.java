package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.scalars.activations.ActivationFunction;

import java.util.LinkedList;
import java.util.List;

/**
 * A {@link Neuron} that caches activations for later retrieval.
 *
 * <p>This neuron extends the base {@link Neuron} class and stores a list of
 * {@link Activation} objects, which contain the input values, target digit,
 * output, and gradient for each activation. This is useful for
 * <strong>backpropagation, debugging, or analysis</strong> of the neuron’s behavior over time.
 *
 * <p>The class is annotated with {@link Decorator} to indicate that it
 * enhances the standard neuron with caching functionality.
 *
 * @see Neuron
 * @see ActivationFunction
 */
@Decorator
public class ActivationsCachedNeuron extends Neuron {

    /**
     * Represents a single activation of a neuron.
     *
     * <p>Stores the following information for each activation:
     * <ul>
     *     <li>{@code inputs} – the input values received by this neuron during activation</li>
     *     <li>{@code output} – the output produced by the neuron</li>
     *     <li>{@code parameterGradients_df_dp} – the gradient calculated for this output (useful for learning)</li>
     * </ul>
     */
    public record Activation(float[] inputs, float output, float[] parameterGradients_df_dp) {

    }

    /**
     * The list of cached activations for this neuron.
     */
    private final List<Activation> cachedActivations = new LinkedList<>();

    /**
     * Constructs a new neuron with caching capabilities.
     *
     * @param name               the name of the neuron
     * @param inputs             the list of input neurons
     * @param activationFunction the activation function used by this neuron
     */
    public ActivationsCachedNeuron(String name, List<? extends Neuron> inputs, ActivationFunction activationFunction) {
        super(name, inputs, activationFunction);
    }

    /**
     * Activates the neuron and caches the result.
     *
     * <p>If the neuron is already activated, this method does nothing.
     * Otherwise, it performs the standard {@link Neuron#activate()} operation
     * and stores an {@link Activation} record in the cache containing:
     * inputs, logit, output, and gradient.
     */
    public synchronized void activate() {
        if (isActivated()) {
            return;
        }
        super.activate();
        float[] inputValues = getInputValues();
        float output = getOutput();
        cachedActivations.add(new Activation(inputValues, output, calculateParameterGradients_df_dp(inputValues)));
    }

    /**
     * Returns the cached activations.
     *
     * @return the list of {@link Activation} records representing past activations
     */
    public List<Activation> getCache() {
        return cachedActivations;
    }

    /**
     * Clears all cached activations.
     *
     * <p>This method is synchronized to ensure thread safety when modifying
     * the cache.
     */
    public synchronized void clearCache() {
        cachedActivations.clear();
    }

}
