package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link Regime} implementation that performs full-batch training.
 *
 * <p>If constructed with a delegate regime, training is forwarded to the
 * delegate. If no delegate is provided (i.e. constructed with {@link #BatchRegime()}),
 * this regime defaults to <em>batch mode</em>: the optimizer is invoked once
 * per training call using the entire {@link DataSet} as a single batch.</p>
 *
 * <h3>Behavior</h3>
 * <ul>
 *   <li><strong>With delegate:</strong> All calls to {@link #train} are
 *   forwarded to the delegate.</li>
 *   <li><strong>Without delegate:</strong> The loss gradient is computed over
 *   the full dataset and the optimizer is applied once.</li>
 * </ul>
 *
 * @see DelegatedRegime
 */
@Strategy(role = Strategy.Role.CONCRETE)
public final class BatchRegime extends DelegatedRegime {

    /**
     * Creates a batch regime with no delegate.
     * <p>In this mode the optimizer is invoked once using the full dataset
     * as a single batch.</p>
     */
    public BatchRegime() {
        this(null);
    }

    /**
     * Creates a batch regime that delegates training to the given regime.
     *
     * @param regime the regime to delegate to, or {@code null} for standalone batch mode
     */
    public BatchRegime(Regime regime) {
        super(regime);
    }

    /**
     * Trains the estimation function either by delegating to the wrapped
     * regime (if non-null) or, in standalone mode, by invoking the optimizer
     * once using the entire dataset as a single batch.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @param objective          the objective (loss) function
     * @param optimizer          the optimizer to apply parameter updates
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        regime.train(estimationFunction, trainingSet, objective, optimizer);
    }
}
