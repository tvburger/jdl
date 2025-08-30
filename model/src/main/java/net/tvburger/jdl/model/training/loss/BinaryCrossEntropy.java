package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Floats;

/**
 * Binary Cross-Entropy (BCE) loss function.
 * <p>
 * This loss function is used for binary classification tasks
 * where each output neuron predicts the probability of a binary label
 * (0 or 1) using a {@code sigmoid} activation function.
 * It computes the cross-entropy between the predicted probability
 * and the target label.
 * </p>
 *
 * <h3>Definition</h3>
 * For a single output:
 * <pre>
 *     E = - [ y * log(a) + (1 - y) * log(1 - a) ]
 * </pre>
 * where:
 * <ul>
 *   <li>{@code y} is the true label in {0,1}</li>
 *   <li>{@code a} is the predicted probability (sigmoid output)</li>
 * </ul>
 *
 * <h3>Gradients</h3>
 * The derivative of the BCE loss with respect to the output activation {@code a} is:
 * <pre>
 *     dE/da = -( y / a - (1 - y) / (1 - a) )
 * </pre>
 * When combined with the derivative of the sigmoid activation {@code f'(z) = a(1 - a)},
 * this simplifies to the familiar error signal:
 * <pre>
 *     δ = a - y
 * </pre>
 *
 * <h3>Usage</h3>
 * <ul>
 *   <li>Output layer: one sigmoid unit per target</li>
 *   <li>Loss function: {@code BinaryCrossEntropy}</li>
 *   <li>Training: use the error signals δ = a - y as the starting point for backpropagation</li>
 * </ul>
 *
 * <h3>Numerical stability</h3>
 * To avoid {@code log(0)} or division by zero, a small epsilon (e.g. 1e-7)
 * should be added when clamping predicted probabilities {@code a} into (0,1).
 */
@Strategy(Strategy.Role.CONCRETE)
public class BinaryCrossEntropy implements DimensionLossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateDimensionLoss(float estimated, float target) {
        float a = clamp01(estimated); // the activated value = estimated = predicted
        float y = target; // the target output value
        // Binary cross-entropy for this output
        return -(y * (float) Math.log(a + Floats.EPSILON)
                + (1f - y) * (float) Math.log(1f - a + Floats.EPSILON));
    }

    /**
     * {@inheritDoc}
     */
    public float determineGradient(float estimated, float target) {
        float a = clamp01(estimated);
        float y = target;
        // dE/da for binary cross-entropy
        float dE_da = -(y / (a + Floats.EPSILON) - (1f - y) / (1f - a + Floats.EPSILON));
        return dE_da;
    }

    private static float clamp01(float f) {
        if (f < Floats.EPSILON) return Floats.EPSILON;
        if (f > 1f - Floats.EPSILON) return 1f - Floats.EPSILON;
        return f;
    }

}
