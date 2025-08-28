package net.tvburger.jdl.model.learning;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * A {@link Trainer} implementation that performs <em>batch training</em>,
 * where the entire dataset is provided to the wrapped trainer in a single call.
 * </p>
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>Validates that the dataset is compatible with the given
 *       estimation function before training begins.</li>
 *   <li>Delegates directly to the wrapped {@link Trainer}, passing the
 *       entire dataset at once.</li>
 *   <li>No splitting, iteration, or batching is performed—this is
 *       the simplest training mode.</li>
 * </ul>
 *
 * <h2>Use cases</h2>
 * <ul>
 *   <li>Training models with full-dataset gradient descent (batch
 *       gradient descent).</li>
 *   <li>Wrapping existing trainers to ensure dataset compatibility
 *       checks are enforced consistently.</li>
 *   <li>Serving as the baseline strategy in a hierarchy of training
 *       styles (batch vs. mini-batch vs. online).</li>
 * </ul>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * Trainer<NeuralNetwork> backpropTrainer = new BackpropTrainer();
 * Trainer<NeuralNetwork> batchTrainer = new BatchTrainer<>(backpropTrainer);
 *
 * batchTrainer.train(network, fullTrainingData);
 * }</pre>
 *
 * @param <E> the type of estimation function being trained
 */
@Strategy(role = Strategy.Role.CONCRETE)
public final class BatchTrainer<E extends EstimationFunction> implements Trainer<E> {

    private final Trainer<E> trainer;

    /**
     * Creates a new batch trainer that delegates to the given trainer.
     *
     * @param trainer the trainer to use for full-dataset training
     */
    public BatchTrainer(Trainer<E> trainer) {
        this.trainer = trainer;
    }

    /**
     * {@inheritDoc}
     * <p>
     * Validates dataset compatibility and delegates training of the
     * entire dataset to the wrapped trainer.
     * </p>
     *
     * @throws IllegalArgumentException if the dataset is incompatible
     *                                  with the estimation function
     */
    @Override
    public void train(E estimationFunction, DataSet trainingSet) {
        if (!trainingSet.isCompatibleWith(estimationFunction)) {
            throw new IllegalArgumentException("Incompatible data set!");
        }
        trainer.train(estimationFunction, trainingSet);
    }

}
