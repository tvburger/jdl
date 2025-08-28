package net.tvburger.jdl.model.learning;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * A {@link Trainer} decorator that performs training over multiple epochs.
 * <p>
 * This trainer delegates the actual training logic to another {@link Trainer}
 * but repeats the training process for a configurable number of iterations
 * (epochs). An epoch is one complete pass over the entire training dataset.
 * </p>
 *
 * <h2>Responsibilities:</h2>
 * <ul>
 *   <li>Wrap an existing {@link Trainer} implementation.</li>
 *   <li>Control how many times the underlying trainer sees the full dataset.</li>
 *   <li>Expose the number of epochs as a configurable property.</li>
 * </ul>
 */
@Decorator
public class EpochTrainer<E extends EstimationFunction> implements Trainer<E> {

    private final Trainer<E> trainer;
    private int epochs;

    /**
     * Creates a new {@code EpochTrainer} that delegates to the given trainer
     * and defaults to 1 epoch.
     *
     * @param trainer the underlying trainer, must not be {@code null}
     */
    public EpochTrainer(Trainer<E> trainer) {
        this(trainer, 1);
    }


    /**
     * Creates a new {@code EpochTrainer} that delegates to the given trainer
     * and repeats training for the specified number of epochs.
     *
     * @param trainer the underlying trainer, must not be {@code null}
     * @param epochs  the number of epochs (must be >= 1)
     */
    public EpochTrainer(Trainer<E> trainer, int epochs) {
        this.trainer = trainer;
        this.epochs = epochs;
    }

    /**
     * Returns the number of epochs configured for this trainer.
     *
     * @return the number of epochs
     */
    public final int getEpochs() {
        return epochs;
    }

    /**
     * Sets the number of epochs to use when training.
     *
     * @param epochs the number of epochs (must be >= 1)
     */
    public final void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    /**
     * Trains the given estimation function for the configured number of epochs.
     * Each epoch represents a full pass over the provided training set.
     *
     * @param estimationFunction the estimation function to be trained
     * @param trainingSet        the dataset used for training
     */
    public final void train(E estimationFunction, DataSet trainingSet) {
        for (int i = 0; i < epochs; i++) {
            trainer.train(estimationFunction, trainingSet);
        }
    }

}
