package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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
public class SummedError<N extends Number> implements SampleLossFunction<N>, BatchLossFunction<N> {

    private final JavaNumberTypeSupport<N> typeSupport;

    public SummedError(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateBatchLoss(List<N> sampleLosses) {
        N loss = typeSupport.zero();
        for (N sampleLoss : sampleLosses) {
            loss = typeSupport.add(loss, sampleLoss);
        }
        return loss;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dJ_dL(int batchSize) {
        return typeSupport.one();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateSampleLoss(List<N> dimensionLosses) {
        N loss = typeSupport.zero();
        for (N dimensionLoss : dimensionLosses) {
            loss = typeSupport.add(loss, dimensionLoss);
        }
        return loss;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dL_dl(int dimensions) {
        return typeSupport.one();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return typeSupport;
    }
}
