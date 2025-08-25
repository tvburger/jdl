package net.tvburger.dlp.nn.activations;

import net.tvburger.dlp.nn.ActivationFunction;

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
