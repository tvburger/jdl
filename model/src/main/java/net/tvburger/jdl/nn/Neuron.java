package net.tvburger.jdl.nn;

import net.tvburger.jdl.nn.activations.Activations;

import java.util.Arrays;
import java.util.List;

public class Neuron {

    protected String name;
    protected List<Neuron> inputs;
    protected float[] storedInputs;
    protected int totalActivations;
    protected float[] weights;
    protected float bias;
    protected float logit;
    protected ActivationFunction activationFunction;
    protected float output;
    protected boolean activated;

    public Neuron(String name, List<Neuron> inputs) {
        this(name, inputs, Activations.reLU());
    }

    public Neuron(String name, List<Neuron> inputs, ActivationFunction activationFunction) {
        this.name = name;
        this.inputs = inputs == null ? List.of() : inputs;
        this.weights = new float[this.inputs.size()];
        this.storedInputs = new float[this.inputs.size()];
        this.activationFunction = activationFunction;
    }

    public String getName() {
        return name;
    }

    public synchronized void activate() {
        if (isActivated()) {
            return;
        }
        logit = bias;
        for (int i = 0; i < inputs.size(); i++) {
            float input = inputs.get(i).getOutput();
            logit += input * weights[i];
            storedInputs[i] += input;
        }
        output = activationFunction.activate(logit);
        totalActivations++;
        activated = true;
    }

    public boolean isActivated() {
        return activated;
    }

    public synchronized void deactivate() {
        activated = false;
    }

    public synchronized void reset() {
        Arrays.fill(storedInputs, 0.0f);
        totalActivations = 0;
    }

    public float[] getStoredInputs() {
        return storedInputs;
    }

    public int getTotalActivations() {
        return totalActivations;
    }

    public float[] getWeights() {
        return weights;
    }

    public void setWeights(float[] weights) {
        this.weights = weights;
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public float getLogit() {
        if (!isActivated()) {
            throw new IllegalStateException("Neuron not activated!");
        }
        return logit;
    }

    public float getOutput() {
        if (!isActivated()) {
            throw new IllegalStateException("Neuron not activated!");
        }
        return output;
    }

    public Float findWeight(Neuron source) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i) == source) {
                return weights[i];
            }
        }
        return null;
    }

    @Override
    public String toString() {
        return name + "{" + activated + ", " + logit + ", " + output + "}" + Arrays.toString(weights) + "+" + bias;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
