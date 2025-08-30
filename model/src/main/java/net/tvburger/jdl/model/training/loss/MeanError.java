package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;

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
    public float calculateBatchLoss(List<Pair<float[], float[]>> batch, Pair<SampleLossFunction, List<DimensionLossFunction>> lossFunctions) {
        float loss = 0.0f;
        for (Pair<float[], float[]> sample : batch) {
            loss += lossFunctions.left().calculateSampleLoss(sample, lossFunctions.right());
        }
        return loss / batch.size();
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
        return loss / dimension;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] determineGradients(float[] estimated, float[] target, List<DimensionLossFunction> lossFunctions) {
        int dimension = estimated.length;
        float[] gradients = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            gradients[i] = get(lossFunctions, i).determineGradient(estimated[i], target[i]) / dimension;
        }
        return gradients;
    }

}
