package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

/**
 * Represents a loss function that operates at the batch level,
 * i.e., across multiple samples. A batch is typically a collection
 * of samples used together in training for efficiency and stability
 * of gradient-based optimization.
 *
 * <p>Implementations of this interface compute both the total batch loss
 * and the gradient of the batch loss with respect to per-sample losses.</p>
 *
 * <p>Definitions:</p>
 * <ul>
 *   <li><b>Batch loss:</b> the aggregated loss over all samples in the batch.</li>
 *   <li><b>Gradient dJ/dL:</b> the derivative of the batch loss with respect to
 *       the per-sample losses, used for backpropagation.</li>
 * </ul>
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface BatchLossFunction<N extends Number> extends LossFunction<N> {

    /**
     * Calculates the total loss for the batch by aggregating the per-sample losses.
     *
     * @param sampleLosses a list of individual sample losses
     * @return the total batch loss
     */
    N calculateBatchLoss(List<N> sampleLosses);

    /**
     * Computes the gradient of the batch loss with respect to the per-sample losses.
     * This is commonly denoted as dJ/dL in backpropagation equations.
     *
     * @param batchSize the number of samples in the batch
     * @return the gradient of the batch loss with respect to each sample loss
     */
    N calculateGradient_dJ_dL(int batchSize);

}
