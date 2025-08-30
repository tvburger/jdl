package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link Regime} decorator that performs training using <em>mini-batches</em>.
 *
 * <p>This regime splits the {@link DataSet} into smaller contiguous subsets
 * (mini-batches) of a configurable size. It then delegates training to the
 * wrapped regime once per mini-batch. This provides a compromise between
 * full-batch training (using the entire dataset at once) and online (stochastic)
 * training (using a single sample at a time).</p>
 *
 * <h3>Behavior</h3>
 * <ul>
 *   <li>The dataset is divided into sequential chunks of size
 *   {@link #getSamplesPerLearning()}.</li>
 *   <li>For each chunk, the delegate {@link Regime} is invoked with that subset.</li>
 *   <li>If the dataset size is not divisible by the batch size, the last
 *   mini-batch will contain the remaining samples.</li>
 * </ul>
 *
 * @see DelegatedRegime
 */
@Strategy(role = Strategy.Role.CONCRETE)
public final class MiniBatchRegime extends DelegatedRegime {

    private int samplesPerLearning;

    /**
     * Creates a new mini-batch trainer with the given mini-batch size.
     *
     * @param samplesPerLearning the number of samples per mini-batch (must be > 0)
     */
    public MiniBatchRegime(int samplesPerLearning) {
        this(null, samplesPerLearning);
    }

    /**
     * Creates a new mini-batch trainer with the given mini-batch size.
     *
     * @param samplesPerLearning the number of samples per mini-batch (must be > 0)
     */
    public MiniBatchRegime(Regime regime, int samplesPerLearning) {
        super(regime);
        this.samplesPerLearning = samplesPerLearning;
    }

    /**
     * Returns the configured number of samples per mini-batch.
     *
     * @return the mini-batch size
     */
    public int getSamplesPerLearning() {
        return samplesPerLearning;
    }

    /**
     * Sets the number of samples per mini-batch.
     *
     * @param samplesPerLearning the new mini-batch size (must be > 0)
     */
    public void setSamplesPerLearning(int samplesPerLearning) {
        this.samplesPerLearning = samplesPerLearning;
    }

    /**
     * Trains the given estimation function by dividing the dataset into
     * mini-batches and delegating training to the wrapped regime for each batch.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to split into mini-batches
     * @param objective          the objective (loss) function
     * @param optimizer          the optimizer to apply updates
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        int offset = 0;
        int trainingSetSize = trainingSet.samples().size();
        do {
            int newOffset = Math.min(trainingSetSize, offset + samplesPerLearning);
            regime.train(estimationFunction, trainingSet.subset(offset, newOffset), objective, optimizer);
            offset = newOffset;
        } while (offset >= trainingSet.samples().size());
    }
}
