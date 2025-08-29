package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

@Strategy(role = Strategy.Role.CONCRETE)
public final class EpochRegime extends DelegatedRegime {

    private int epochs;

    /**
     * Creates a new {@code EpochTrainer} that delegates to the given trainer
     * and defaults to 1 epoch.
     */
    public EpochRegime(Regime regime) {
        this(regime, 1);
    }


    /**
     * Creates a new {@code EpochTrainer} that delegates to the given trainer
     * and repeats training for the specified number of epochs.
     *
     * @param regime
     * @param epochs the number of epochs (must be >= 1)
     */
    public EpochRegime(Regime regime, int epochs) {
        super(regime);
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

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        for (int i = 0; i < epochs; i++) {
            regime.train(estimationFunction, trainingSet, objective, optimizer);
        }
    }
}
