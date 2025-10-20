package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.scalars.AffineTransformation;
import net.tvburger.jdl.model.scalars.NeuronFunction;
import net.tvburger.jdl.model.scalars.activations.ActivationFunction;
import net.tvburger.jdl.model.scalars.activations.Activations;

import java.util.List;

/**
 * A specialized {@link Neuron} that remembers the input values
 * from its most recent activation.
 * <p>
 * This class extends the basic neuron by capturing and storing the
 * raw outputs of its input neurons during the forward pass. The stored
 * inputs do not affect the computation of the neuron’s output but
 * provide additional visibility or state for downstream processing,
 * such as training algorithms, debugging, or interpretability tools.
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>On {@link #activate()}, performs the standard neuron computation
 *       (weighted sum + bias → activation function) and also stores a copy
 *       of the input values provided by connected neurons.</li>
 *   <li>The stored inputs can be retrieved via {@link #getStoredInput(int)}.</li>
 * </ul>
 */
@Decorator
public class LastInputStoredNeuron extends Neuron {

    public static LastInputStoredNeuron create(String name, List<? extends Neuron> inputNodes) {
        return create(name, inputNodes, Activations.linear());
    }

    public static LastInputStoredNeuron create(String name, List<? extends Neuron> inputNodes, ActivationFunction activationFunction) {
        return new LastInputStoredNeuron(name, inputNodes, new NeuronFunction(AffineTransformation.create(inputNodes.size(), JavaNumberTypeSupport.FLOAT), activationFunction));
    }

    private final float[] storedInputs;

    /**
     * Constructs a neuron that stores the inputs from its most recent activation.
     *
     * @param name           human-readable identifier
     * @param inputs         input neurons feeding into this neuron
     * @param neuronFunction the neuron function to compute a forward signal
     */
    public LastInputStoredNeuron(String name, List<? extends Neuron> inputs, NeuronFunction neuronFunction) {
        super(name, inputs, neuronFunction);
        storedInputs = new float[inputs == null ? 0 : inputs.size()];
    }

    /**
     * Activates this neuron by performing the normal weighted sum
     * and activation function, and additionally storing the raw input
     * values from connected neurons.
     */
    public synchronized void activate() {
        super.activate();
        for (int i = 0; i < getInputNodes().size(); i++) {
            storedInputs[i] = getInputNodes().get(i).getOutput();
        }
    }

    /**
     * Returns the input value that was stored for a specific dimension during the
     * most recent {@link #activate()} call.
     * <p>
     * Dimension indexing is 1-based, where {@code d = 1} refers to the first
     * input neuron, {@code d = 2} to the second, and so on.
     *
     * @param d the input dimension index (1-based)
     * @return the stored input value for the specified dimension
     */
    public float getStoredInput(int d) {
        return storedInputs[d - 1];
    }

}
