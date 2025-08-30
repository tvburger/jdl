package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * Represents a specialized loss function that operates on a single dimension
 * of a predicted versus target value. Implementations of this interface define
 * how the "dimension loss" and corresponding gradient are computed.
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface DimensionLossFunction extends LossFunction {

    /**
     * Calculates the loss for a single dimension, given the estimated value
     * and the target value.
     *
     * @param estimated the predicted or estimated value
     * @param target    the expected or target value
     * @return the loss for this dimension as a {@code float}
     */
    float calculateDimensionLoss(float estimated, float target);

    /**
     * Determines the gradient of the loss function with respect to the
     * estimated value, for a single dimension.
     *
     * @param estimated the predicted or estimated value
     * @param target    the expected or target value
     * @return the gradient as a {@code float}
     */
    float determineGradient(float estimated, float target);

}
