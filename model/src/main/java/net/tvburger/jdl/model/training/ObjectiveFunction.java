package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.model.training.loss.BatchLossFunction;
import net.tvburger.jdl.model.training.loss.DimensionLossFunction;
import net.tvburger.jdl.model.training.loss.LossFunction;
import net.tvburger.jdl.model.training.loss.SampleLossFunction;

import java.util.Arrays;
import java.util.List;

/**
 * Represents an objective function that extends the concept of a
 * {@link LossFunction} to provide both loss aggregation and gradient
 * computation.
 *
 * <p>
 * An {@code ObjectiveFunction} defines how the overall loss is calculated
 * across a batch of samples, as well as how the gradients are determined
 * for optimization algorithms (e.g., stochastic gradient descent).
 * </p>
 *
 * <p>
 * Unlike {@link DimensionLossFunction}, {@link SampleLossFunction}, or
 * {@link BatchLossFunction}, which focus on specific granularities,
 * an objective function typically provides the entry point for a training
 * process, combining loss evaluation and gradient calculation in one place.
 * </p>
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface ObjectiveFunction extends LossFunction {

    /**
     * Calculates the loss for a batch of samples.
     *
     * <p>
     * Each sample in the batch is represented as a {@link Pair} of
     * {@code float[]} arrays, where the left element is the estimated
     * (predicted) values and the right element is the target
     * (expected) values. Implementations should define how individual
     * sample losses are aggregated into a single scalar loss for the batch.
     * </p>
     *
     * @param batch a list of samples, where each sample is a pair of
     *              {@code float[]} arrays (estimated vs target)
     * @return the aggregated loss value for the batch
     */
    float calculateLoss(List<Pair<float[], float[]>> batch);

    /**
     * Determines the gradients of the objective function with respect to
     * the estimated values, for a single sample.
     *
     * <p>
     * Implementations should return a gradient array that corresponds in
     * length and order to the {@code estimated} array, such that each entry
     * can be used to update model parameters during optimization.
     * </p>
     *
     * @param samples   the total number of samples in the batch
     * @param estimated the predicted values for a sample
     * @param target    the expected target values for the sample
     * @return an array of gradients, one per dimension of the input
     */
    float[] calculateGradient_dJ_da(int samples, float[] estimated, float[] target);

    /**
     * Creates an {@link ObjectiveFunction} that minimizes the loss
     * according to the given loss function components.
     *
     * <p>
     * This factory method composes together a {@link BatchLossFunction},
     * a {@link SampleLossFunction}, and one or more
     * {@link DimensionLossFunction}s into a concrete
     * {@link ObjectiveFunction} implementation. The resulting objective
     * function calculates dimension-level errors, aggregates them into
     * sample losses, and then aggregates those sample losses into a
     * batch-level loss suitable for optimization.
     * </p>
     *
     * <p>
     * The intent of this method is to provide a convenient entry point
     * for constructing custom objective functions when the optimization
     * goal is loss minimization (as is typical in regression and
     * classification problems).
     * </p>
     *
     * @param batchLossFunction      the function used to aggregate losses
     *                               across the entire batch
     * @param sampleLossFunction     the function used to aggregate losses
     *                               within a single sample
     * @param dimensionLossFunctions one or more functions used to compute
     *                               loss at the individual dimension level
     * @return a composed {@link ObjectiveFunction} that minimizes loss
     */
    static ObjectiveFunction minimize(BatchLossFunction batchLossFunction, SampleLossFunction sampleLossFunction, DimensionLossFunction... dimensionLossFunctions) {
        return new ObjectiveFunctionImpl(batchLossFunction, sampleLossFunction, Arrays.asList(dimensionLossFunctions));
    }

}
