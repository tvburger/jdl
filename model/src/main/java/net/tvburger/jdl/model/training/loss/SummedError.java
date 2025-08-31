package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

/**
 * A loss function implementation that computes the sum of errors
 * rather than their mean.
 *
 * <p>
 * {@code SummedError} can operate at both the sample level
 * ({@link SampleLossFunction}) and the batch level
 * ({@link BatchLossFunction}). Instead of averaging errors,
 * it aggregates them by summation, producing a larger value as the
 * number of dimensions or samples increases.
 * </p>
 *
 * <p>
 * This approach is useful in contexts where the absolute magnitude of
 * the error across dimensions or samples is important, or where mean
 * aggregation would dilute the contribution of individual errors.
 * </p>
 *
 * <p>
 * Typical usage includes situations where cumulative loss across all
 * outputs or samples is desired, rather than a normalized mean value.
 */
@Strategy(Strategy.Role.CONCRETE)
public class SummedError implements SampleLossFunction, BatchLossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateBatchLoss(List<Float> sampleLosses) {
        float loss = 0.0f;
        for (float sampleLoss : sampleLosses) {
            loss += sampleLoss;
        }
        return loss;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dJ_dL(int batchSize) {
        return 1.0f;
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
        return loss;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dL_dl(int dimensions) {
        return 1.0f;
    }

}
