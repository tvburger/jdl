package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;

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
@Strategy(role = Strategy.Role.CONCRETE)
public class SummedError implements SampleLossFunction, BatchLossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateBatchLoss(List<Pair<float[], float[]>> batch, Pair<SampleLossFunction, List<DimensionLossFunction>> lossFunctions) {
        float loss = 0.0f;
        for (Pair<float[], float[]> sample : batch) {
            loss += lossFunctions.left().calculateSampleLoss(sample, lossFunctions.right());
        }
        return loss;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateSampleLoss(Pair<float[], float[]> sample, List<DimensionLossFunction> lossFunctions) {
        float loss = 0.0f;
        float[] estimated = sample.left();
        float[] target = sample.right();
        int dimension = estimated.length;
        for (int i = 0; i < dimension; i++) {
            loss += get(lossFunctions, i).calculateDimensionLoss(estimated[i], target[i]);
        }
        return loss;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] determineGradients(float[] estimated, float[] target, List<DimensionLossFunction> lossFunctions) {
        int dimension = estimated.length;
        float[] gradients = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            gradients[i] = get(lossFunctions, i).determineGradient(estimated[i], target[i]);
        }
        return gradients;
    }

}
