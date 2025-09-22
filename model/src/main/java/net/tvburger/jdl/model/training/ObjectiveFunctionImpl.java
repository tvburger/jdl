package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.Mediator;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.model.training.loss.BatchLossFunction;
import net.tvburger.jdl.model.training.loss.DimensionLossFunction;
import net.tvburger.jdl.model.training.loss.SampleLossFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * The implementation of the {@link ObjectiveFunction}.
 *
 * <p>
 * {@code ObjectiveFunctionImpl} composes together different levels of
 * loss functions — dimension, sample, and batch — to provide a complete
 * objective function suitable for training and optimization.
 * </p>
 *
 * <p>
 * The implementation delegates responsibility as follows:
 * <ul>
 *   <li>Uses a {@link DimensionLossFunction} to compute per-dimension error
 *       and gradients.</li>
 *   <li>Uses a {@link SampleLossFunction} to aggregate dimension-level losses
 *       into a single sample loss.</li>
 *   <li>Uses a {@link BatchLossFunction} to aggregate sample losses into a
 *       batch loss.</li>
 * </ul>
 * </p>
 */
@Mediator
@Strategy(Strategy.Role.CONCRETE)
public class ObjectiveFunctionImpl implements ObjectiveFunction {

    private final SampleLossFunction sampleLossFunction;
    private final List<DimensionLossFunction> dimensionLossFunctions;
    private final BatchLossFunction batchLossFunction;

    /**
     * Creates an {@code ObjectiveFunctionImpl} that computes losses at all levels:
     * dimension, sample, and batch.
     *
     * <p>This constructor links together the components needed for hierarchical
     * loss computation:</p>
     * <ul>
     *     <li>{@link BatchLossFunction}: computes the total loss over a batch of samples.</li>
     *     <li>{@link SampleLossFunction}: computes the loss for a single sample by
     *         aggregating per-dimension losses.</li>
     *     <li>{@link DimensionLossFunction}: computes the loss for each individual
     *         output dimension.</li>
     * </ul>
     *
     * @param batchLossFunction      the loss function for the batch level
     * @param sampleLossFunction     the loss function for the sample level
     * @param dimensionLossFunctions the list of loss functions for each output dimension
     */
    public ObjectiveFunctionImpl(BatchLossFunction batchLossFunction, SampleLossFunction sampleLossFunction, List<DimensionLossFunction> dimensionLossFunctions) {
        this.batchLossFunction = batchLossFunction;
        this.sampleLossFunction = sampleLossFunction;
        this.dimensionLossFunctions = dimensionLossFunctions;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateLoss(List<Pair<Float[], Float[]>> batch) {
        List<Float> sampleLosses = new ArrayList<>(batch.size());
        List<Float> dimensionLosses = new ArrayList<>();
        for (Pair<Float[], Float[]> sample : batch) {
            Float[] estimated = sample.left();
            Float[] target = sample.right();
            int dimensions = estimated.length;
            for (int d = 0; d < dimensions; d++) {
                dimensionLosses.add(getDimensionLossFunction(d).calculateDimensionLoss(estimated[d], target[d]));
            }
            sampleLosses.add(sampleLossFunction.calculateSampleLoss(dimensionLosses));
            dimensionLosses.clear();
        }
        return batchLossFunction.calculateBatchLoss(sampleLosses);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Float[] calculateGradient_dJ_da(int batchSize, Float[] estimated, Float[] target) {
        int dimensions = target.length;
        Float[] gradients = new Float[dimensions];
        for (int d = 0; d < dimensions; d++) {
            gradients[d] = batchLossFunction.calculateGradient_dJ_dL(batchSize)
                    * sampleLossFunction.calculateGradient_dL_dl(dimensions)
                    * getDimensionLossFunction(d).calculateGradient_dl_da(estimated[d], target[d]);
        }
        return gradients;
    }

    private DimensionLossFunction getDimensionLossFunction(int d) {
        return dimensionLossFunctions.get(d % dimensionLossFunctions.size());
    }
}
