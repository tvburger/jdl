package net.tvburger.jdl.model.nn.activations;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * Concrete {@link ActivationFunction} implementing the logistic sigmoid:
 *
 * <pre>
 * σ(x) = 1 / (1 + e^{-x})
 * </pre>
 *
 * <p>The sigmoid squashes any real-valued input into the (0, 1) interval and is
 * commonly used for binary classification logits and hidden-layer activations.</p>
 */
@Strategy(Strategy.Role.CONCRETE)
public class Sigmoid implements ActivationFunction {

    /**
     * Applies the logistic sigmoid to the given logit.
     *
     * @param logit input value (pre-activation)
     * @return σ(logit) in (0, 1)
     */
    @Override
    public float activate(float logit) {
        return 1.0f / (1.0f + (float) Math.exp(-logit));
    }

    /**
     * Returns the gradient of the sigmoid at the given <em>output</em> value.
     * <p>
     * Given {@code output = σ(x)}, the derivative is {@code output * (1 - output)}.
     * This avoids recomputing the activation and is numerically stable.
     *
     * @param output the sigmoid output value y = σ(x), expected in [0, 1]
     * @return the gradient dσ/dx evaluated at the corresponding input x
     * @throws IllegalArgumentException if {@code output} is NaN or infinite
     */
    @Override
    public float determineGradientForOutput(float output) {
        if (!Float.isFinite(output)) {
            throw new IllegalArgumentException("Output must be a finite number.");
        }
        // Clamp to [0,1] if callers might pass slightly out-of-range values due to numeric drift.
        float properOutput = Math.min(1.0f, Math.max(0.0f, output));
        return properOutput * (1.0f - properOutput);
    }
}
