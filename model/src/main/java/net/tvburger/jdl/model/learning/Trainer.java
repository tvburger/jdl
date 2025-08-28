package net.tvburger.jdl.model.learning;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * Strategy interface for training an {@link EstimationFunction} using
 * a given {@link DataSet}.
 * <p>
 * This interface represents the <b>training strategy</b> in a machine
 * learning or statistical modeling context. Implementations define
 * how an estimation function (e.g., a classifier, regressor, or predictor)
 * should be fitted to training data.
 * </p>
 *
 * <h2>Responsibilities:</h2>
 * <ul>
 *   <li>Encapsulate the logic for fitting a model to data.</li>
 *   <li>Allow different training algorithms to be plugged in interchangeably
 *       by following the Strategy pattern.</li>
 *   <li>Operate on a provided {@link EstimationFunction} instance, updating
 *       its internal parameters based on the supplied training set.</li>
 * </ul>
 */
@DomainObject
@Strategy(role = Strategy.Role.INTERFACE)
public interface Trainer<E extends EstimationFunction> {

    /**
     * Trains the given estimation function using the provided training set.
     *
     * @param estimationFunction the estimation function to be trained,
     *                           must not be {@code null}
     * @param trainingSet        the training data used to fit the model,
     *                           must not be {@code null}
     * @throws IllegalArgumentException if either parameter is {@code null}
     */
    void train(E estimationFunction, DataSet trainingSet);

}
