package net.tvburger.dlp.nn;

public class InputNeuron extends Neuron {

    public InputNeuron(String name) {
        super(name, null);
    }

    public void setInputValue(float input) {
        logit = input;
        output = input;
        activated = true;
    }

    @Override
    public synchronized void activate() {
        if (!isActivated()) {
            throw new IllegalStateException("Must set input value!");
        }
    }
}
