package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * A {@link DimensionLossFunction} that computes the squared error
 * between an estimated value and its target value.
 *
 * <p>
 * The squared error is defined as:
 * <pre>
 * SE = (estimated - target)^2
 * </pre>
 * </p>
 *
 * <p>
 * This loss function penalizes larger deviations more heavily than
 * smaller ones, making it widely used in regression problems and as
 * the basis of the Mean Squared Error (MSE) objective.
 * </p>
 *
 * <p>
 * When combined with a {@link ScaledError} (e.g., with a factor of
 * {@code 0.5}), it produces the conventional MSE formulation:
 * <pre>
 * MSE = 0.5 * (estimated - target)^2
 * </pre>
 */
@Strategy(Strategy.Role.CONCRETE)
public class SquaredError implements DimensionLossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateDimensionLoss(float estimated, float target) {
        return (target - estimated) * (target - estimated);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float determineGradient(float estimated, float target) {
        return 2 * (target - estimated);
    }

}
