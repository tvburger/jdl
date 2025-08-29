package net.tvburger.jdl.model.nn.activations;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * This activation function, also known as Identity or "no-activation" just returns the logit as output.
 */
@Strategy(role = Strategy.Role.CONCRETE)
public class Linear implements ActivationFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float activate(float logit) {
        return logit;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float determineGradientForOutput(float output) {
        return 1;
    }

}
