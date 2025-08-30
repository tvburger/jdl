package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;

import java.util.List;

/**
 * Represents a loss function that operates at the batch level,
 * i.e., across multiple samples. A batch is typically a collection
 * of samples used together in training for efficiency and stability
 * of gradient-based optimization.
 *
 * <p>
 * A {@code BatchLossFunction} leverages a {@link SampleLossFunction}
 * together with a set of {@link DimensionLossFunction}s to compute
 * the overall loss for an entire batch of samples.
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface BatchLossFunction extends LossFunction {

    /**
     * Calculates the total loss for a batch of samples.
     *
     * <p>
     * Each sample in the batch is represented as a {@link Pair} of
     * float arrays, where the left element is the estimated (predicted)
     * values and the right element is the target (expected) values.
     * The provided {@link SampleLossFunction} and associated list of
     * {@link DimensionLossFunction}s are used to compute the loss for
     * each sample, which is then aggregated into the batch loss.
     * </p>
     *
     * @param batch         a list of samples, where each sample is a pair
     *                      of {@code float[]} arrays (estimated vs target)
     * @param lossFunctions a pair consisting of a {@link SampleLossFunction}
     *                      (for per-sample loss calculation) and the list
     *                      of {@link DimensionLossFunction}s used within
     *                      that sample loss calculation
     * @return the aggregated batch loss as a {@code float}
     */
    float calculateBatchLoss(List<Pair<float[], float[]>> batch, Pair<SampleLossFunction, List<DimensionLossFunction>> lossFunctions);

}
