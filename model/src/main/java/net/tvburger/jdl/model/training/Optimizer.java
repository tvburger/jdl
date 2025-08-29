package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * Defines an optimization algorithm for training an {@link EstimationFunction}.
 *
 * <p>
 * An {@code Optimizer} updates the parameters of an estimation function
 * (e.g., a machine learning model) to minimize an {@link ObjectiveFunction}
 * when applied to a given {@link DataSet}.
 * </p>
 *
 * <p>
 * Different optimizers implement different update rules (e.g., gradient descent,
 * stochastic gradient descent, Adam). This interface abstracts those details
 * to allow interchangeable use of optimizers in training regimes.
 * </p>
 *
 * @param <E> the type of estimation function being optimized
 * @see EstimationFunction
 * @see ObjectiveFunction
 * @see DataSet
 */
@DomainObject
@Strategy(role = Strategy.Role.INTERFACE)
public interface Optimizer<E extends EstimationFunction> {

    /**
     * An {@link Optimizer} variant that is restricted to online (sample-by-sample)
     * optimization.
     *
     * <p>
     * Implementations of {@code OnlineOnly} optimize the estimation function one
     * sample at a time rather than processing the entire dataset in a batch.
     * This is common in streaming or online learning contexts.
     * </p>
     *
     * <p>
     * The default {@link #optimize(EstimationFunction, DataSet, ObjectiveFunction)}
     * implementation iterates through the dataset and delegates to
     * {@link #optimize(EstimationFunction, DataSet.Sample, ObjectiveFunction)} for
     * each sample.
     * </p>
     *
     * @param <E> the type of estimation function being optimized
     */
    @Strategy(role = Strategy.Role.CONCRETE)
    interface OnlineOnly<E extends EstimationFunction> extends Optimizer<E> {

        /**
         * Default implementation of batch optimization for {@code OnlineOnly}
         * optimizers. Iterates over the dataset and delegates optimization to
         * the per-sample method.
         *
         * @param estimationFunction the estimation function (model) to optimize
         * @param trainingSet        the dataset used for optimization
         * @param objective          the objective function that defines the loss
         *                           and gradients
         */
        @Override
        default void optimize(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective) {
            trainingSet.forEach(s -> optimize(estimationFunction, s, objective));
        }

        /**
         * Optimizes the given estimation function using a single sample
         * and the specified objective function.
         *
         * <p>
         * Implementations typically calculate the gradient of the loss function
         * for the given sample and update the parameters of the estimation function
         * immediately, enabling online or stochastic optimization.
         * </p>
         *
         * @param estimationFunction the estimation function (model) to optimize
         * @param sample             the training sample used for optimization
         * @param objective          the objective function that defines the loss
         *                           and gradients
         */
        void optimize(E estimationFunction, DataSet.Sample sample, ObjectiveFunction objective);

    }

    /**
     * Optimizes the given estimation function based on the entire training set
     * and the specified objective function.
     *
     * <p>
     * Typical implementations evaluate the {@link ObjectiveFunction} across the
     * training set, compute gradients, and update the parameters of the
     * estimation function accordingly.
     * </p>
     *
     * @param estimationFunction the estimation function (model) to optimize
     * @param trainingSet        the dataset used for optimization
     * @param objective          the objective function that defines the loss
     *                           and gradients
     */
    void optimize(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective);

}
