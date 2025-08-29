package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.nn.activations.ActivationFunction;

import java.util.LinkedList;
import java.util.List;

@Decorator
public class ActivationsCachedNeuron extends Neuron {

    public record Activation(float[] inputs, float digit, float output, float gradient) {

    }

    private final List<Activation> cachedActivations = new LinkedList<>();

    public ActivationsCachedNeuron(String name, List<? extends Neuron> inputs, ActivationFunction activationFunction) {
        super(name, inputs, activationFunction);
    }

    public synchronized void activate() {
        if (isActivated()) {
            return;
        }
        super.activate();
        float[] inputs = new float[getInputs().size()];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] += getInputs().get(i).getOutput();
        }
        cachedActivations.add(new Activation(inputs, getLogit(), getOutput(), getActivationFunction().determineGradientForOutput(getOutput())));
    }

    public List<Activation> getCache() {
        return cachedActivations;
    }

    public synchronized void clearCache() {
        cachedActivations.clear();
    }

}
