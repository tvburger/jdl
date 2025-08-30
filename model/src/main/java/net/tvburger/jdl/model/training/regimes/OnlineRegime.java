package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link Regime} decorator that performs <em>online training</em>
 * (also known as stochastic training).
 *
 * <p>In online mode, the training set is traversed one sample at a time.
 * For each individual sample, the delegate {@link Regime} is invoked with a
 * dataset subset of size one. This provides the highest update frequency,
 * at the cost of noisier gradients compared to batch or mini-batch training.</p>
 *
 * <h3>Behavior</h3>
 * <ul>
 *   <li>If constructed with a delegate, training is forwarded to it for each
 *   single-sample subset.</li>
 *   <li>If constructed without a delegate, the default regime provided by
 *   {@link DelegatedRegime} is used, which directly calls the
 *   {@link Optimizer} once per sample.</li>
 * </ul>
 *
 * @see DelegatedRegime
 */
@Strategy(Strategy.Role.CONCRETE)
public final class OnlineRegime extends DelegatedRegime {

    /**
     * Creates a new online regime with no delegate.
     * <p>In this mode the optimizer is invoked once per sample directly.</p>
     */
    public OnlineRegime() {
        this(null);
    }

    /**
     * Creates a new online regime that delegates to the given regime.
     *
     * @param regime the underlying regime to delegate to, or {@code null} for standalone online mode
     */
    public OnlineRegime(Regime regime) {
        super(regime);
    }

    /**
     * Trains the given estimation function in online mode.
     * <p>
     * Each sample of the training set is wrapped in a singleton subset,
     * and the delegate regime is invoked once per sample.
     * </p>
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to iterate over sample by sample
     * @param objective          the objective (loss) function
     * @param optimizer          the optimizer to apply updates
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        for (int i = 0; i < trainingSet.samples().size(); i++) {
            regime.train(estimationFunction, trainingSet.subset(i, i + 1), objective, optimizer);
        }
    }

}
