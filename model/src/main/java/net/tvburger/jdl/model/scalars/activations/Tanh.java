package net.tvburger.jdl.model.scalars.activations;

import net.tvburger.jdl.common.patterns.Strategy;

@Strategy(Strategy.Role.CONCRETE)
public class Tanh implements ActivationFunction {

    @Override
    public float activate(float logit) {
        // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        return (float) Math.tanh(logit);
    }

    @Override
    public float determineGradientForOutput(float output) {
        if (!Float.isFinite(output)) {
            throw new IllegalArgumentException("Output must be a finite number.");
        }
        // derivative of tanh(x) with respect to x is 1 - tanh(x)^2
        return 1.0f - output * output;
    }
}
