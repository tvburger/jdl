package net.tvburger.dlp.activations;

import net.tvburger.dlp.ActivationFunction;

public class ReLU implements ActivationFunction {

    @Override
    public float activate(float logit) {
        return Math.max(0, logit);
    }

}
