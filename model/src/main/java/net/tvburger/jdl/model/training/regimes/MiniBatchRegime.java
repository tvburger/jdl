package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link TrainingRegime} implementation that performs <em>mini-batch training</em>,
 * where the training dataset is divided into smaller batches of a fixed size,
 * and the wrapped trainer is invoked once per batch.
 * <p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>The dataset is partitioned sequentially into contiguous mini-batches,
 *       each containing at most {@code samplesPerLearning} samples.</li>
 *   <li>The wrapped {@code trainer} is invoked on each mini-batch in order.</li>
 *   <li>If the dataset size is not divisible by the mini-batch size, the
 *       final batch will contain the remaining samples.</li>
 *   <li>A dataset compatibility check is performed before training starts;
 *       an {@link IllegalArgumentException} is thrown if the dataset is
 *       not compatible with the estimation function.</li>
 * </ul>
 *
 * <h2>Use cases</h2>
 * <ul>
 *   <li>Training neural networks or other models with stochastic gradient
 *       descent variants that rely on mini-batch updates.</li>
 *   <li>Balancing convergence stability (batch training) with computational
 *       efficiency and generalization (online training).</li>
 *   <li>Reusing existing {@link TrainingRegime} implementations by delegating
 *       batch-wise work to them.</li>
 * </ul>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * // Wrap an existing batch trainer in a mini-batch strategy
 * Trainer<NeuralNetwork> backpropTrainer = new BackpropTrainer();
 * Trainer<NeuralNetwork> miniBatchTrainer =
 *         new MiniBatchTrainer<>(backpropTrainer, 32) { };
 *
 * miniBatchTrainer.train(network, trainingData);
 * }</pre>
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
