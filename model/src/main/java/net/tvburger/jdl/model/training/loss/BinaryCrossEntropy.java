package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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
public class BinaryCrossEntropy<N extends Number> implements DimensionLossFunction<N> {

    private final JavaNumberTypeSupport<N> typeSupport;

    public BinaryCrossEntropy(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateDimensionLoss(N estimated, N target) {
        N a = typeSupport.clamp01(estimated); // predicted probability
        N y = target; // true label

        // a + epsilon
        N aPlusEps = typeSupport.add(a, typeSupport.epsilon());
        // 1 - a
        N oneMinusA = typeSupport.subtract(typeSupport.one(), a);
        // 1 - y
        N oneMinusY = typeSupport.subtract(typeSupport.one(), y);
        // 1 - a + epsilon
        N oneMinusAPlusEps = typeSupport.add(oneMinusA, typeSupport.epsilon());

        // log(a + epsilon)
        N logA = typeSupport.log(aPlusEps);
        // log(1 - a + epsilon)
        N logOneMinusA = typeSupport.log(oneMinusAPlusEps);

        // y * log(a + epsilon)
        N yLogA = typeSupport.multiply(y, logA);
        // (1 - y) * log(1 - a + epsilon)
        N oneMinusYLogOneMinusA = typeSupport.multiply(oneMinusY, logOneMinusA);

        // sum
        N sum = typeSupport.add(yLogA, oneMinusYLogOneMinusA);

        // -sum
        return typeSupport.negate(sum);
    }

    /**
     * {@inheritDoc}
     */
    public N calculateGradient_dl_da(N estimated, N target) {
        N a = typeSupport.clamp01(estimated);
        N y = target;
        // a + epsilon
        N aPlusEps = typeSupport.add(a, typeSupport.epsilon());
        // 1 - a
        N oneMinusA = typeSupport.subtract(typeSupport.one(), a);
        // 1 - a + epsilon
        N oneMinusAPlusEps = typeSupport.add(oneMinusA, typeSupport.epsilon());
        // 1 - y
        N oneMinusY = typeSupport.subtract(typeSupport.one(), y);

        // y / (a + epsilon)
        N yOverA = typeSupport.divide(y, aPlusEps);
        // (1 - y) / (1 - a + epsilon)
        N oneMinusYOverOneMinusA = typeSupport.divide(oneMinusY, oneMinusAPlusEps);
        // y / (a + epsilon) - (1 - y) / (1 - a + epsilon)
        N diff = typeSupport.subtract(yOverA, oneMinusYOverOneMinusA);
        // -(...)
        return typeSupport.negate(diff);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return typeSupport;
    }
}
