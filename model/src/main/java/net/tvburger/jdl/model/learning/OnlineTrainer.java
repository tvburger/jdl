package net.tvburger.jdl.model.learning;

import net.tvburger.jdl.common.patterns.Adapter;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * A {@link Trainer} implementation that performs
 * <em>online training</em> (stochastic gradient descent style),
 * where the estimation function is updated once per sample.
 * <p>
 *
 * <h2>Use cases</h2>
 * <ul>
 *   <li>Training estimation functions incrementally using stochastic
 *       gradient descent or other sample-wise update algorithms.</li>
 *   <li>Wrapping existing batch trainers for online use by decomposing
 *       each sample into a dataset of size one.</li>
 *   <li>Providing a convenient functional API for sample-wise training
 *       logic via {@link Function#asTrainer()}.</li>
 * </ul>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * // Functional style: define how to train on a single sample
 * OnlineTrainer.Function<LinearModel> fn = (model, sample) -> {
 *     float prediction = model.predict(sample.getFeatures());
 *     float error = sample.getTargets()[0] - prediction;
 *     model.updateWeights(sample.getFeatures(), error, 0.01f);
 * };
 *
 * Trainer<LinearModel> trainer = fn.asTrainer();
 * trainer.train(linearModel, dataSet);
 *
 * // Wrapping another Trainer as online
 * Trainer<LinearModel> batchTrainer = new BatchTrainerImpl();
 * Trainer<LinearModel> online = new OnlineTrainer<>(batchTrainer);
 * online.train(linearModel, dataSet);
 * }</pre>
 *
 * @param <E> the type of estimation function being trained
 */
@Strategy(role = Strategy.Role.CONCRETE)
public final class OnlineTrainer<E extends EstimationFunction> implements Trainer<E> {

    /**
     * Functional interface for training an estimation function on a single sample.
     *
     * @param <E> the type of estimation function being trained
     */
    @Strategy(role = Strategy.Role.INTERFACE)
    public interface Function<E extends EstimationFunction> {

        /**
         * Trains the given estimation function using a single sample.
         *
         * @param estimationFunction the estimation function to update
         * @param sample             the sample to train on
         */
        void train(E estimationFunction, DataSet.Sample sample);

        /**
         * Wraps this sample-wise training function into a full {@link Trainer}
         * that can process entire datasets by applying it to each sample.
         *
         * @return an {@link OnlineTrainer} delegating to this function
         */
        @Adapter
        default OnlineTrainer<E> asTrainer() {
            return new OnlineTrainer<>(this);
        }
    }

    private final Function<E> trainSample;

    /**
     * Wraps an existing {@link Trainer} as an {@code OnlineTrainer}
     * by decomposing its training sets into single-sample datasets.
     *
     * @param trainer the trainer to wrap
     */
    @Adapter
    public OnlineTrainer(Trainer<E> trainer) {
        this((Function<E>) (e, s) -> trainer.train(e, DataSet.of(s)));
    }

    /**
     * Constructs an {@code OnlineTrainer} directly from a sample-wise
     * training function.
     *
     * @param trainSample the sample-wise training logic
     */
    public OnlineTrainer(Function<E> trainSample) {
        this.trainSample = trainSample;
    }

    /**
     * {@inheritDoc}
     * <p>
     * Iterates over each sample in the dataset and invokes the
     * configured {@link Function} to update the estimation function.
     * </p>
     *
     * @throws IllegalArgumentException if the dataset is incompatible
     *                                  with the estimation function
     */
    public void train(E estimationFunction, DataSet trainingSet) {
        if (!trainingSet.isCompatibleWith(estimationFunction)) {
            throw new IllegalArgumentException("Incompatible data set!");
        }
        trainingSet.forEach(s -> trainSample.train(estimationFunction, s));
    }

}
