package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

/**
 * Represents a loss function that operates at the per-sample level.
 * <p>
 * A sample loss aggregates the losses of individual output dimensions
 * (dimension losses, {@code l}) into a single per-sample loss (L),
 * which can then be used to compute batch-level loss (J) in
 * {@link BatchLossFunction}.
 * </p>
 *
 * <p>Definitions:</p>
 * <ul>
 *   <li><b>Sample loss (L):</b> the aggregated loss for a single training sample.</li>
 *   <li><b>Gradient dL/dl:</b> the derivative of the sample loss with respect to each
 *       dimension loss, used for backpropagation from the sample to individual dimensions.</li>
 * </ul>
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface SampleLossFunction extends LossFunction {

    /**
     * Calculates the total loss for a single sample by aggregating
     * the losses of its dimensions.
     *
     * @param dimensionLosses a list of losses for each output dimension
     * @return the total sample loss
     */
    float calculateSampleLoss(List<Float> dimensionLosses);

    /**
     * Computes the gradient of the sample loss with respect to the
     * individual dimension losses. This corresponds to dL/dl in
     * backpropagation equations.
     *
     * @param dimensions the number of output dimensions
     * @return the gradient of the sample loss with respect to each dimension loss
     */
    float calculateGradient_dL_dl(int dimensions);

}
