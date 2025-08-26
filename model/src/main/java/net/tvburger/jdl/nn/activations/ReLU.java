package net.tvburger.jdl.nn.activations;

import net.tvburger.jdl.nn.ActivationFunction;

public class ReLU implements ActivationFunction {

    @Override
    public String name() {
        return "ReLU";
    }

    @Override
    public float activate(float logit) {
        return Math.max(0, logit);
    }

}
