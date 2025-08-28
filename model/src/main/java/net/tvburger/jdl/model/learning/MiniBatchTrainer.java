package net.tvburger.jdl.model.learning;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * A {@link Trainer} implementation that performs <em>mini-batch training</em>,
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
 *   <li>Reusing existing {@link Trainer} implementations by delegating
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
 *
 * @param <E> the type of estimation function being trained
 */
@Strategy(role = Strategy.Role.CONCRETE)
public final class MiniBatchTrainer<E extends EstimationFunction> implements Trainer<E> {

    private final Trainer<E> trainer;
    private int samplesPerLearning;

    /**
     * Creates a new mini-batch trainer with the given mini-batch size.
     *
     * @param trainer            the trainer to use for the mini-batches
     * @param samplesPerLearning the number of samples per mini-batch (must be > 0)
     */
    public MiniBatchTrainer(Trainer<E> trainer, int samplesPerLearning) {
        this.trainer = trainer;
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
     * {@inheritDoc}
     * </p>
     * Splits the training set into mini-batches and invokes the wrapped
     * {@link Trainer} once per batch.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @throws IllegalArgumentException if the dataset is incompatible with the model
     */
    @Override
    public void train(E estimationFunction, DataSet trainingSet) {
        if (!trainingSet.isCompatibleWith(estimationFunction)) {
            throw new IllegalArgumentException("Incompatible data set!");
        }
        int offset = 0;
        int trainingSetSize = trainingSet.samples().size();
        do {
            int newOffset = Math.min(trainingSetSize, offset + samplesPerLearning);
            trainer.train(estimationFunction, trainingSet.subset(offset, newOffset));
            offset = newOffset;
        } while (offset >= trainingSet.samples().size());
    }

}
