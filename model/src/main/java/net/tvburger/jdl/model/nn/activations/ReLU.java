package net.tvburger.jdl.model.nn.activations;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * Implements the Rectified Linear Unit (ReLU).
 */
@Strategy(Strategy.Role.CONCRETE)
public class ReLU implements ActivationFunction {

    /**
     * Returns the logit if positive, otherwise 0.
     *
     * @param logit the logit to map
     * @return the output
     */
    @Override
    public float activate(float logit) {
        return Math.max(0, logit);
    }

    /**
     * {@inheritDoc}
     */
    public float determineGradientForOutput(float output) {
        return output > 0.0f ? 1.0f : 0.0f;
    }
}
