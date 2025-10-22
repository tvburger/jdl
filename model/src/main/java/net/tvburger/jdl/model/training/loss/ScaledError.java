package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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
public class ScaledError<N extends Number> implements DimensionLossFunction<N>, SampleLossFunction<N>, BatchLossFunction<N> {

    private final JavaNumberTypeSupport<N> typeSupport;
    private final N scale;
    private final LossFunction<N> lossFunction;

    /**
     * Creates a new {@code ScaledError} that decorates the given
     * {@link LossFunction} with a scaling factor.
     *
     * @param scale        the factor by which to scale the computed loss values
     * @param lossFunction the underlying {@link LossFunction} to decorate
     */
    public ScaledError(JavaNumberTypeSupport<N> typeSupport, N scale, LossFunction<N> lossFunction) {
        this.typeSupport = typeSupport;
        this.scale = scale;
        this.lossFunction = lossFunction;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateBatchLoss(List<N> sampleLosses) {
        return typeSupport.multiply(((BatchLossFunction<N>) lossFunction).calculateBatchLoss(sampleLosses), scale);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dJ_dL(int batchSize) {
        return typeSupport.multiply(((BatchLossFunction<N>) lossFunction).calculateGradient_dJ_dL(batchSize), scale);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateDimensionLoss(N estimated, N target) {
        return typeSupport.multiply(((DimensionLossFunction<N>) lossFunction).calculateDimensionLoss(estimated, target), scale);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dl_da(N estimated, N target) {
        return typeSupport.multiply(((DimensionLossFunction<N>) lossFunction).calculateGradient_dl_da(estimated, target), scale);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateSampleLoss(List<N> dimensionLosses) {
        return typeSupport.multiply(((SampleLossFunction<N>) lossFunction).calculateSampleLoss(dimensionLosses), scale);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateGradient_dL_dl(int dimensions) {
        return typeSupport.multiply(((SampleLossFunction<N>) lossFunction).calculateGradient_dL_dl(dimensions), scale);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }
}
