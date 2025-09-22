package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A training regime that performs <strong>online training</strong>.
 * <p>
 * In online training (also known as stochastic training), the model
 * is updated after processing each individual sample, rather than
 * waiting for a full batch or mini-batch. This can lead to faster
 * convergence in some cases, but updates are noisier compared to
 * batch or mini-batch regimes.
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>The training set is traversed sequentially.</li>
 *   <li>For each sample, a single-sample subset is created and passed
 *       to the optimizer.</li>
 *   <li>The optimizer performs an update immediately after each sample.</li>
 * </ul>
 *
 * <h2>Use cases</h2>
 * <p>
 * Online regimes are useful for streaming data, very large datasets
 * that cannot fit in memory, or when responsiveness to new data
 * is important.
 * </p>
 *
 * @see Regime
 * @see BatchRegime
 * @see MiniBatchRegime
 */
@Strategy(Strategy.Role.CONCRETE)
public final class OnlineRegime implements Regime {

    /**
     * Trains the given estimation function in an online manner by iterating
     * over each sample in the dataset and invoking the optimizer immediately
     * for each one.
     *
     * @param estimationFunction the model or function to train
     * @param trainingSet        the dataset containing training samples
     * @param objective          the objective/loss function
     * @param optimizer          the optimizer used to update parameters
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction<Float>> void train(E estimationFunction, DataSet<Float> trainingSet, ObjectiveFunction objective, Optimizer<? super E, Float> optimizer) {
        for (int i = 0; i < trainingSet.size(); i++) {
            optimizer.optimize(estimationFunction, trainingSet.subset(i, i + 1), objective);
        }
    }

}
