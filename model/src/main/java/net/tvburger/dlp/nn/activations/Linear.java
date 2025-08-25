package net.tvburger.dlp.nn.activations;

import net.tvburger.dlp.nn.ActivationFunction;

public class Linear implements ActivationFunction {

    @Override
    public String name() {
        return "linear";
    }

    @Override
    public float activate(float logit) {
        return logit;
    }

}
