package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * Defines a training regime for machine learning models.
 *
 * <p>
 * A {@code Regime} encapsulates the strategy used to train an
 * {@link EstimationFunction} (such as a model or predictor) on a
 * {@link DataSet} using a specified {@link ObjectiveFunction} and
 * {@link Optimizer}.
 * </p>
 *
 * <p>
 * Implementations of this interface define the control flow of
 * training — for example, how many epochs to run, how to split data
 * into batches, or when to apply optimization steps. This allows
 * experimentation with different training strategies without changing
 * the underlying model, objective, or optimizer.
 * </p>
 *
 * @see EstimationFunction
 * @see DataSet
 * @see ObjectiveFunction
 * @see Optimizer
 */
@DomainObject
@Strategy(role = Strategy.Role.INTERFACE)
public interface Regime {

    /**
     * Trains the given {@link EstimationFunction} on the provided
     * {@link DataSet} according to the specified
     * {@link ObjectiveFunction} and {@link Optimizer}.
     *
     * <p>
     * The training process typically involves iteratively evaluating
     * the objective function on the training set, calculating gradients,
     * and applying the optimizer to update the estimation function's
     * parameters.
     * </p>
     *
     * @param <E>                the type of estimation function being trained
     * @param estimationFunction the model or estimation function to be trained
     * @param trainingSet        the dataset used for training
     * @param objective          the objective function that defines the
     *                           training loss and gradients
     * @param optimizer          the optimizer used to update the estimation
     *                           function’s parameters based on gradients
     */
    <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer);

}
