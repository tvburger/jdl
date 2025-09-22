package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

/**
 * A training regime that performs <strong>mini-batch training</strong>.
 * <p>
 * In mini-batch training, the dataset is split into contiguous batches
 * of a fixed size, and each batch is trained sequentially using the
 * delegated {@link BatchRegime}.
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>The batch size is configurable via {@link BatchSizeConfigurable}.</li>
 *   <li>The dataset is partitioned into consecutive mini-batches of that size.</li>
 *   <li>Each mini-batch is trained independently by delegating to
 *       {@link BatchRegime}.</li>
 *   <li>No shuffling is performed; batches are taken in dataset order.</li>
 * </ul>
 *
 * <h2>Use cases</h2>
 * <p>
 * Mini-batch regimes combine the computational efficiency of batch training
 * with the convergence stability of stochastic updates, making them the most
 * commonly used training regime in practice.
 * </p>
 *
 * @see BatchRegime
 * @see BatchSizeConfigurable
 */
@Strategy(Strategy.Role.CONCRETE)
public final class MiniBatchRegime extends DelegatedRegime implements BatchSizeConfigurable {

    /**
     * Creates a new mini-batch regime with the specified mini-batch size.
     *
     * @param batchSize the number of samples per mini-batch (must be &gt; 0)
     */
    public MiniBatchRegime(int batchSize) {
        super(new BatchRegime());
        setHyperparameter(HP_BATCH_SIZE, batchSize);
    }

    /**
     * Trains the given estimation function by splitting the dataset into
     * mini-batches of size {@link #getBatchSize()} and delegating each
     * batch to the underlying {@link BatchRegime}.
     *
     * @param estimationFunction the model or function to train
     * @param trainingSet        the dataset containing training samples
     * @param objective          the objective/loss function
     * @param optimizer          the optimizer used to update parameters
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction<Float>> void train(E estimationFunction, DataSet<Float> trainingSet, ObjectiveFunction objective, Optimizer<? super E, Float> optimizer) {
        int offset = 0;
        int trainingSetSize = trainingSet.size();
        do {
            int newOffset = Math.min(trainingSetSize, offset + getBatchSize());
            regime.train(estimationFunction, trainingSet.subset(offset, newOffset), objective, optimizer);
            offset = newOffset;
        } while (offset >= trainingSet.size());
    }
}
