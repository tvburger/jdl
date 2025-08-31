package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

/**
 * A decorator loss function that applies a fixed scaling factor
 * to the results of another loss function.
 *
 * <p>
 * {@code ScaledError} can operate at multiple levels of loss
 * calculation — dimension, sample, and batch — by delegating to
 * an underlying {@link LossFunction} and scaling its results by
 * a constant multiplier.
 * </p>
 *
 * <p>
 * For example, when wrapping a {@link SquaredError} with a scaling
 * factor of {@code 0.5}, this class produces the commonly used
 * Mean Squared Error (MSE) formulation:
 * <pre>
 * MSE = 0.5 * (predicted - target)^2
 * </pre>
 */
@Decorator
@Strategy(Strategy.Role.CONCRETE)
public class ScaledError implements DimensionLossFunction, SampleLossFunction, BatchLossFunction {

    private final float scale;
    private final LossFunction lossFunction;

    /**
     * Creates a new {@code ScaledError} that decorates the given
     * {@link LossFunction} with a scaling factor.
     *
     * @param scale        the factor by which to scale the computed loss values
     * @param lossFunction the underlying {@link LossFunction} to decorate
     */
    public ScaledError(float scale, LossFunction lossFunction) {
        this.scale = scale;
        this.lossFunction = lossFunction;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateBatchLoss(List<Float> sampleLosses) {
        return ((BatchLossFunction) lossFunction).calculateBatchLoss(sampleLosses) * scale;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dJ_dL(int batchSize) {
        return ((BatchLossFunction) lossFunction).calculateGradient_dJ_dL(batchSize) * scale;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateDimensionLoss(float estimated, float target) {
        return ((DimensionLossFunction) lossFunction).calculateDimensionLoss(estimated, target) * scale;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dl_da(float estimated, float target) {
        return ((DimensionLossFunction) lossFunction).calculateGradient_dl_da(estimated, target) * scale;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateSampleLoss(List<Float> dimensionLosses) {
        return ((SampleLossFunction) lossFunction).calculateSampleLoss(dimensionLosses) * scale;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateGradient_dL_dl(int dimensions) {
        return ((SampleLossFunction) lossFunction).calculateGradient_dL_dl(dimensions) * scale;
    }
}
