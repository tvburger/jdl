package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.model.HyperparameterConfigurable;

/**
 * Marker interface for components that expose a configurable learning rate.
 */
public interface EpochConfigurable extends HyperparameterConfigurable {

    /**
     * The hyperparameter key name for epochs.
     */
    String HP_EPOCHS = "epochs";

    /**
     * Returns the configured number of training epochs.
     *
     * @return the number of epochs (must be ≥ 1)
     */
    default int getEpochs() {
        return getHyperparameter(HP_EPOCHS, Integer.class);
    }

    /**
     * Sets the number of training epochs.
     *
     * @param epochs the number of epochs (must be ≥ 1)
     * @throws IllegalArgumentException if {@code epochs < 1}
     */
    default void setEpochs(int epochs) {
        if (epochs < 1) {
            throw new IllegalArgumentException();
        }
        setHyperparameter(HP_EPOCHS, epochs);
    }

}