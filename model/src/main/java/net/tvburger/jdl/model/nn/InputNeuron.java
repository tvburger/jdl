package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.DomainObject;

/**
 * Represents an input neuron in a neural network.
 * <p>
 * Unlike hidden or output neurons, an {@code InputNeuron} does not
 * compute its output from weighted inputs and an activation function.
 * Instead, its output value is set directly from outside the network
 * (e.g., by feeding in features of a training or inference sample).
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>{@link #setInputValue(float)} assigns the fixed output value
 *       for this input neuron and marks it as activated.</li>
 *   <li>{@link #activate()} does not perform any computation; it merely
 *       verifies that the neuron has been given an input value, otherwise
 *       it throws an {@link IllegalStateException}.</li>
 *   <li>{@link #deactivate()} (inherited) clears the activation flag so
 *       a new input can be assigned in a subsequent pass.</li>
 * </ul>
 *
 * <h2>Usage example</h2>
 * <pre>{@code
 * InputNeuron n1 = new InputNeuron("x1");
 * InputNeuron n2 = new InputNeuron("x2");
 *
 * n1.setInputValue(0.5f);
 * n2.setInputValue(-1.2f);
 *
 * // these are now ready to be consumed by downstream neurons
 * float v1 = n1.getOutput(); // 0.5
 * float v2 = n2.getOutput(); // -1.2
 * }</pre>
 *
 * <p>
 * Input neurons are typically the entry points of a network, one per
 * feature dimension in the dataset.
 * </p>
 */
@DomainObject
public class InputNeuron extends Neuron {

    private float input;

    /**
     * Constructs a new input neuron with the given name.
     * Input neurons have no inputs or activation function.
     *
     * @param name human-readable identifier
     */
    public InputNeuron(String name) {
        super(name, null, null);
    }

    @Override
    public Float estimateScalar(Float[] inputs) {
        return input;
    }

    /**
     * Sets the externally provided input value for this neuron and marks
     * it as activated.
     *
     * @param input the input value to assign
     */
    public void setInputValue(float input) {
        this.input = input;
        activate();
    }

    /**
     * {@inheritDoc}
     */
    public float getOutput() {
        return input;
    }

}
