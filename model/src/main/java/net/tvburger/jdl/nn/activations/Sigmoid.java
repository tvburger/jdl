package net.tvburger.jdl.nn.activations;

import net.tvburger.jdl.nn.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public String name() {
        return "sigmoid";
    }

    @Override
    public float activate(float logit) {
        return 1.0f / (1.0f + (float) Math.exp(-logit));
    }
}
