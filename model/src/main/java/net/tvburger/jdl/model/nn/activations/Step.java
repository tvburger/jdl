package net.tvburger.jdl.model.nn.activations;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * Implements the step activation. For a given threshold will return a low value (e.g. -1 or 0), and when
 * the logit is above or equals the threshold return the high value (by default 1).
 * <p>
 * Note: this function is non-differentiable at the threshold, and otherwise 0. Therefor we don't support
 * to determine a gradient as it can't be used for detecting any optimization.
 */
@Strategy(Strategy.Role.CONCRETE)
public class Step implements ActivationFunction {

    private float lowValue = 0.0f;
    private float highValue = 1.0f;
    private float threshold = 0.0f;

    /**
     * Get the low value that is returned when the logit is below the threshold. Default is 0.
     *
     * @return the low value
     */
    public float getLowValue() {
        return lowValue;
    }

    /**
     * Set the low value for this step.
     *
     * @param lowValue the low value
     */
    public void setLowValue(float lowValue) {
        this.lowValue = lowValue;
    }

    /**
     * Get the high value that is returned when the logit is equal or above the threshold. Default is 1.
     *
     * @return the high value
     */
    public float getHighValue() {
        return highValue;
    }

    /**
     * Set the high value for this step
     *
     * @param highValue the high value
     */
    public void setHighValue(float highValue) {
        this.highValue = highValue;
    }

    /**
     * Get the threshold of this step
     *
     * @return the threshold
     */
    public float getThreshold() {
        return threshold;
    }

    /**
     * Set the threshold of this step
     *
     * @param threshold the threshold
     */
    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    /**
     * Returns the low value if the logit is below the threshold, otherwise the high value
     *
     * @param logit the logit to map
     * @return the low or high value depending on the threshold
     */
    @Override
    public float activate(float logit) {
        return logit < threshold ? lowValue : highValue;
    }

}
