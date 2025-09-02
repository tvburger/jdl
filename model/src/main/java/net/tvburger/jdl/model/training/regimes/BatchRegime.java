package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A training regime that performs classic <strong>batch training</strong>.
 * <p>
 * In batch training, the optimizer is applied once to the entire training set
 * in a single call. This is equivalent to full-batch gradient descent, where
 * all samples are used together to compute gradients and update parameters.
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>Delegates directly to {@link Optimizer#optimize} with the provided
 *       estimation function, dataset, and objective.</li>
 *   <li>No batching or shuffling is performed; the optimizer receives the
 *       complete dataset.</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * This regime is useful for small datasets where processing the entire batch
 * in memory is feasible. For larger datasets or stochastic training, see
 * mini-batch or online regimes.
 *
 * @see Regime
 * @see Optimizer
 */
@Strategy(Strategy.Role.CONCRETE)
public final class BatchRegime implements Regime {

    /**
     * Trains the given estimation function using the full training set in one step.
     *
     * @param estimationFunction the model or function being trained
     * @param trainingSet        the dataset containing all training samples
     * @param objective          the objective/loss function
     * @param optimizer          the optimizer that updates the function parameters
     * @param <E>                the type of estimation function being trained
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        optimizer.optimize(estimationFunction, trainingSet, objective);
    }
}
