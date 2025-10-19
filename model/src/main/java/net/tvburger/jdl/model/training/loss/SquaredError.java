package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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
public class SquaredError<N extends Number> implements DimensionLossFunction<N> {

    private final JavaNumberTypeSupport<N> typeSupport;

    public SquaredError(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateDimensionLoss(N estimated, N target) {
        N error = typeSupport.subtract(estimated, target);
        return typeSupport.multiply(error, error);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dl_da(N estimated, N target) {
        N error = typeSupport.subtract(estimated, target);
        return typeSupport.multiply(error, 2);
    }

    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return typeSupport;
    }
}
