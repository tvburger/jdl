package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.nn.activations.ActivationFunction;

import java.util.Arrays;
import java.util.List;

/**
 * A specialized {@link Neuron} that accumulates the values of its inputs
 * across multiple activations.
 * <p>
 * Unlike a standard neuron, which only produces an output for the current
 * forward pass, a {@code SummedInputStoredNeuron} maintains a running sum
 * of the raw outputs of its input neurons. This allows the neuron to track
 * aggregate input activity over time in addition to computing its normal
 * output through the weighted sum + bias + activation function.
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>On {@link #activate()}, performs the standard neuron computation
 *       and additionally updates {@code summedStoredInputs} by adding the
 *       current outputs of its input neurons to the running totals.</li>
 *   <li>The accumulated sums can be retrieved with
 *       {@link #getSummedStoredInputs()}.</li>
 *   <li>The sums can be reset with {@link #clearStoredInputs()}.</li>
 * </ul>
 *
 * <h2>Use cases</h2>
 * <ul>
 *   <li>Tracking cumulative input activity across epochs or mini-batches.</li>
 *   <li>Supporting training algorithms that rely on aggregated statistics.</li>
 *   <li>Debugging or analysis of input signal distributions.</li>
 * </ul>
 */
@Decorator
public class SummedInputStoredNeuron extends Neuron {

    private final float[] summedStoredInputs;

    /**
     * Constructs a neuron that stores cumulative input values
     * across activations.
     *
     * @param name               human-readable identifier
     * @param inputs             input neurons feeding into this neuron
     * @param activationFunction activation function to apply
     */
    public SummedInputStoredNeuron(String name, List<? extends Neuron> inputs, ActivationFunction activationFunction) {
        super(name, inputs, activationFunction);
        summedStoredInputs = new float[inputs.size()];
    }

    /**
     * Activates this neuron by performing the normal weighted sum and
     * activation, and then accumulating the raw outputs of its input neurons
     * into {@code summedStoredInputs}.
     */
    public synchronized void activate() {
        if (isActivated()) {
            return;
        }
        super.activate();
        for (int i = 0; i < getInputs().size(); i++) {
            summedStoredInputs[i] += getInputs().get(i).getOutput();
        }
    }

    /**
     * Returns the cumulative sums of inputs across all activations since
     * the last call to {@link #clearStoredInputs()}.
     *
     * @return the running totals of input values
     */
    public float[] getSummedStoredInputs() {
        return summedStoredInputs;
    }

    /**
     * Resets all accumulated input sums to zero.
     */
    public synchronized void clearStoredInputs() {
        Arrays.fill(summedStoredInputs, 0.0f);
    }

}
