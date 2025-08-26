package net.tvburger.jdl.nn.activations;

import net.tvburger.jdl.nn.ActivationFunction;

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
