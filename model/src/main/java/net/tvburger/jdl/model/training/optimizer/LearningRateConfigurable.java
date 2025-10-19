package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.model.HyperparameterConfigurable;

/**
 * Marker interface for components that expose a configurable learning rate.
 */
public interface LearningRateConfigurable<N extends Number> extends HyperparameterConfigurable {

    /** The hyperparameter name for learning rate. */
    String HP_LEARNING_RATE = "learningRate";

    /**
     * Returns the current learning rate (step size).
     *
     * @return the learning rate value
     */
    @SuppressWarnings("unchecked")
    default N getLearningRate() {
        return (N) getHyperparameter(HP_LEARNING_RATE);
    }

    /**
     * Sets the learning rate (step size).
     *
     * @param learningRate the new learning rate (must be > 0)
     */
    default void setLearningRate(N learningRate) {
        setHyperparameter(HP_LEARNING_RATE, learningRate);
    }
}