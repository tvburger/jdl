package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.model.HyperparameterConfigurable;

/**
 * Interface for components that expose a configurable training batch size.
 * <p>
 * Batch size controls how many samples are used to compute a single gradient
 * update during training. It is a training hyperparameter (not an optimizer
 * hyperparameter).
 */
public interface BatchSizeConfigurable extends HyperparameterConfigurable {

    /**
     * The hyperparameter name for batch size.
     */
    String HP_BATCH_SIZE = "batchSize";

    /**
     * Returns the configured training batch size.
     *
     * @return the batch size (must be ≥ 1)
     */
    default int getBatchSize() {
        return getHyperparameter(HP_BATCH_SIZE, Integer.class);
    }

    /**
     * Sets the training batch size.
     *
     * @param batchSize the batch size (must be ≥ 1)
     * @throws IllegalArgumentException if {@code batchSize < 1}
     */
    default void setBatchSize(int batchSize) {
        if (batchSize < 1) {
            throw new IllegalArgumentException();
        }
        setHyperparameter(HP_BATCH_SIZE, batchSize);
    }
}
