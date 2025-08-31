package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

/**
 * A loss function implementation that computes the mean error
 * across samples or batches.
 *
 * <p>
 * {@code MeanError} provides a simple averaging strategy for loss
 * calculation. It can operate both at the sample level
 * ({@link SampleLossFunction}) and at the batch level
 * ({@link BatchLossFunction}), making it reusable in different
 * stages of the loss aggregation hierarchy.
 * </p>
 *
 * <p>
 * At the sample level, this class typically averages the individual
 * dimension losses produced by {@link DimensionLossFunction}s.
 * At the batch level, it averages the losses of all samples within
 * the batch.
 * </p>
 *
 * <p>
 * This implementation is often used as a building block in
 * higher-level objective functions such as Mean Squared Error (MSE)
 * or Binary Cross-Entropy (BCE), where mean aggregation is required
 * either per sample, per batch, or both.
 * </p>
 *
 * @see SampleLossFunction
 * @see BatchLossFunction
 * @see DimensionLossFunction
 */
@Strategy(Strategy.Role.CONCRETE)
public class MeanError implements SampleLossFunction, BatchLossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateBatchLoss(List<Float> sampleLosses) {
        float loss = 0.0f;
        for (Float sampleLoss : sampleLosses) {
            loss += sampleLoss;
        }
        return loss / sampleLosses.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dJ_dL(int batchSize) {
        return 1.0f / batchSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateSampleLoss(List<Float> dimensionLosses) {
        float loss = 0.0f;
        for (float dimensionLoss : dimensionLosses) {
            loss += dimensionLoss;
        }
        return loss / dimensionLosses.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dL_dl(int dimensions) {
        return 1.0f / dimensions;
    }

}
