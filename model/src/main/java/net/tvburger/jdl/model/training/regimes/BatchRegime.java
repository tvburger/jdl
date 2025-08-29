package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

@Strategy(role = Strategy.Role.CONCRETE)
public final class BatchRegime extends DelegatedRegime {

    public BatchRegime() {
        this(null);
    }

    public BatchRegime(Regime regime) {
        super(regime);
    }

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        regime.train(estimationFunction, trainingSet, objective, optimizer);
    }
}
