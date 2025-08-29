package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Mediator;
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
@Strategy(role = Strategy.Role.INTERFACE)
public interface ObjectiveFunction extends LossFunction {

    /**
     * Calculates the aggregated loss across a batch of samples.
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
    float calculateAggregatedLoss(List<Pair<float[], float[]>> batch);

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
     * @param estimated the predicted values for a sample
     * @param target    the expected target values for the sample
     * @return an array of gradients, one per dimension of the input
     */
    float[] determineGradients(float[] estimated, float[] target);

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
        return new Impl(batchLossFunction, Pair.of(sampleLossFunction, Arrays.asList(dimensionLossFunctions)));
    }

    /**
     * The implementation of the {@link ObjectiveFunction}.
     *
     * <p>
     * {@code ObjectiveFunctionImpl} composes together different levels of
     * loss functions — dimension, sample, and batch — to provide a complete
     * objective function suitable for training and optimization.
     * </p>
     *
     * <p>
     * The implementation delegates responsibility as follows:
     * <ul>
     *   <li>Uses a {@link DimensionLossFunction} to compute per-dimension error
     *       and gradients.</li>
     *   <li>Uses a {@link SampleLossFunction} to aggregate dimension-level losses
     *       into a single sample loss.</li>
     *   <li>Uses a {@link BatchLossFunction} to aggregate sample losses into a
     *       batch loss.</li>
     * </ul>
     * </p>
     */
    @Mediator
    @Strategy(role = Strategy.Role.CONCRETE)
    class Impl implements ObjectiveFunction {

        private final Pair<SampleLossFunction, List<DimensionLossFunction>> sampleLossFunctions;
        private final BatchLossFunction batchLossFunction;

        private Impl(BatchLossFunction batchLossFunction, Pair<SampleLossFunction, List<DimensionLossFunction>> sampleLossFunctions) {
            this.batchLossFunction = batchLossFunction;
            this.sampleLossFunctions = sampleLossFunctions;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public float calculateAggregatedLoss(List<Pair<float[], float[]>> batch) {
            return batchLossFunction.calculateBatchLoss(batch, sampleLossFunctions);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public float[] determineGradients(float[] estimated, float[] target) {
            return sampleLossFunctions.left().determineGradients(estimated, target, sampleLossFunctions.right());
        }

    }
}
