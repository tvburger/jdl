package net.tvburger.jdl.nn.activations;

import net.tvburger.jdl.nn.ActivationFunction;

public class Step implements ActivationFunction {

    private float lowValue = 0.0f;
    private float highValue = 1.0f;
    private float threshold = 0.0f;

    public float getLowValue() {
        return lowValue;
    }

    public void setLowValue(float lowValue) {
        this.lowValue = lowValue;
    }

    public float getHighValue() {
        return highValue;
    }

    public void setHighValue(float highValue) {
        this.highValue = highValue;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    @Override
    public String name() {
        return "step";
    }

    @Override
    public float activate(float logit) {
        return threshold < 0 ? lowValue : highValue;
    }
}
