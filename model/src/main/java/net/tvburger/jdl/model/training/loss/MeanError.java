package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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
public class MeanError<N extends Number> implements SampleLossFunction<N>, BatchLossFunction<N> {

    private final JavaNumberTypeSupport<N> typeSupport;

    public MeanError(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
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
        return typeSupport.divide(loss, sampleLosses.size());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dJ_dL(int batchSize) {
        return typeSupport.divide(typeSupport.one(), batchSize);
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
        return typeSupport.divide(loss, dimensionLosses.size());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dL_dl(int dimensions) {
        return typeSupport.divide(typeSupport.one(), dimensions);
    }

}
