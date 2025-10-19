package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * Represents a loss function that operates at the per-dimension level.
 * <p>
 * Each output dimension of a sample can have its own loss, which is then
 * aggregated into a per-sample loss (L) or batch loss (J) by higher-level
 * loss functions such as {@link BatchLossFunction}.
 * </p>
 *
 * <p>Definitions:</p>
 * <ul>
 *   <li><b>Dimension loss (l):</b> the loss computed for a single output dimension.</li>
 *   <li><b>Gradient dl/da:</b> the derivative of the dimension loss with respect to the activation (output) of this dimension,
 *       used in backpropagation.</li>
 * </ul>
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface DimensionLossFunction<N extends Number> extends LossFunction<N> {

    /**
     * Calculates the loss for a single dimension, given the estimated value
     * and the target value.
     *
     * @param estimated the predicted or estimated value
     * @param target    the expected or target value
     * @return the loss for this dimension as a {@code float}
     */
    N calculateDimensionLoss(N estimated, N target);

    /**
     * Calculates the gradient of the dimension loss with respect to the
     * activation (output) of this dimension. This corresponds to dl/da
     * in backpropagation formulas.
     *
     * @param estimated the predicted or estimated value for this dimension
     * @param target    the expected or target value for this dimension
     * @return the gradient of the loss with respect to the activation
     */
    N calculateGradient_dl_da(N estimated, N target);

}
