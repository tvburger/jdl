package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.model.HyperparameterConfigurable;

/**
 * Marker interface for components that expose a configurable learning rate.
 */
public interface LearningRateConfigurable extends HyperparameterConfigurable {

    /** The hyperparameter name for learning rate. */
    String HP_LEARNING_RATE = "learningRate";

    /**
     * Returns the current learning rate (step size).
     *
     * @return the learning rate value
     */
    default float getLearningRate() {
        return getHyperparameter(HP_LEARNING_RATE, Float.class);
    }

    /**
     * Sets the learning rate (step size).
     *
     * @param learningRate the new learning rate (must be > 0)
     */
    default void setLearningRate(float learningRate) {
        setHyperparameter(HP_LEARNING_RATE, learningRate);
    }
}