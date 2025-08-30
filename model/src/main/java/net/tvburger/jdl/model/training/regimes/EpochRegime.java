package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link Regime} decorator that repeats training for a fixed number of epochs.
 *
 * <p>This regime wraps another regime and delegates training to it multiple times.
 * Each call to {@link #train} will invoke the delegate regime exactly
 * {@link #getEpochs()} times over the same dataset. This is useful for
 * composing training behaviors where you want to control how many epochs of
 * training are performed independently of other concerns (batching, reporting,
 * early stopping, etc.).</p>
 *
 * <h3>Notes</h3>
 * <ul>
 *   <li>The number of epochs must be at least 1.</li>
 *   <li>This regime is typically used in a {@link net.tvburger.jdl.model.training.regimes.ChainedRegime}
 *   together with other decorators.</li>
 * </ul>
 *
 * @see DelegatedRegime
 */
@Strategy(role = Strategy.Role.CONCRETE)
public final class EpochRegime extends DelegatedRegime {

    private int epochs;

    /**
     * Creates a new {@code EpochRegime} that delegates to the given regime
     * and repeats training for the specified number of epochs.
     *
     * @param regime the underlying regime to delegate training to
     * @param epochs the number of epochs to run (must be {@code >= 1})
     */
    public EpochRegime(Regime regime, int epochs) {
        super(regime);
        this.epochs = epochs;
    }

    /**
     * Returns the number of epochs configured for this regime.
     *
     * @return the number of epochs
     */
    public final int getEpochs() {
        return epochs;
    }

    /**
     * Sets the number of epochs to use when training.
     *
     * @param epochs the number of epochs (must be {@code >= 1})
     */
    public final void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    /**
     * Trains the given estimation function by repeatedly delegating to the
     * wrapped regime for the configured number of epochs.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @param objective          the loss function to evaluate
     * @param optimizer          the optimizer to apply parameter updates
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        for (int i = 0; i < epochs; i++) {
            regime.train(estimationFunction, trainingSet, objective, optimizer);
        }
    }
}
