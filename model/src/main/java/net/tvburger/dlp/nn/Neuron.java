package net.tvburger.dlp.nn;

import net.tvburger.dlp.ActivationFunction;
import net.tvburger.dlp.activations.Activations;

import java.util.List;

public class Neuron {

    protected String name;
    protected List<Neuron> inputs;
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
            logit += inputs.get(i).getOutput() * weights[i];
        }
        output = activationFunction.activate(logit);
        activated = true;
    }

    public boolean isActivated() {
        return activated;
    }

    public void reset() {
        activated = false;
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
}
