package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;

import java.util.List;

/**
 * Represents a loss function that operates at the sample level,
 * i.e., over an entire set of estimated versus target values rather
 * than a single dimension.
 *
 * <p>
 * A {@code SampleLossFunction} coordinates multiple
 * {@link DimensionLossFunction} implementations to calculate the overall
 * sample loss and per-dimension gradients. This is typically used in
 * machine learning contexts where a sample consists of multiple features
 * or output dimensions.
 */
@DomainObject
@Strategy(role = Strategy.Role.INTERFACE)
public interface SampleLossFunction extends LossFunction {

    /**
     * Calculates the total loss for a sample by applying the corresponding
     * {@link DimensionLossFunction} to each element of the estimated and
     * target arrays.
     *
     * @param sample        a pair containing the estimated values (left)
     *                      and target values (right) for a single sample
     * @param lossFunctions the list of dimension-specific loss functions
     *                      to be applied
     * @return the total loss for the sample as a {@code float}
     */
    float calculateSampleLoss(Pair<float[], float[]> sample, List<DimensionLossFunction> lossFunctions);

    /**
     * Determines the gradient of the loss function for each dimension of
     * the sample, given the estimated and target arrays.
     *
     * @param estimated     the predicted values for the sample
     * @param target        the expected target values for the sample
     * @param lossFunctions the list of dimension-specific loss functions
     *                      to be applied
     * @return an array of gradient values, one per dimension
     */
    float[] determineGradients(float[] estimated, float[] target, List<DimensionLossFunction> lossFunctions);

    /**
     * Retrieves the appropriate {@link DimensionLossFunction} for a given
     * index. If the index exceeds the number of available loss functions,
     * it is wrapped around using modulo arithmetic.
     *
     * <p>
     * This provides a convenient way to cycle through loss functions
     * when the number of dimensions exceeds the number of functions.
     * </p>
     *
     * @param lossFunctions the list of dimension-specific loss functions
     * @param index         the index of the dimension
     * @return the corresponding {@link DimensionLossFunction}
     */
    default DimensionLossFunction get(List<DimensionLossFunction> lossFunctions, int index) {
        return lossFunctions.get(index % lossFunctions.size());
    }
}
