package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

@Strategy(role = Strategy.Role.CONCRETE)
public final class OnlineRegime extends DelegatedRegime {

    public OnlineRegime() {
        this(null);
    }

    public OnlineRegime(Regime regime) {
        super(regime);
    }

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        for (int i = 0; i < trainingSet.samples().size(); i++) {
            regime.train(estimationFunction, trainingSet.subset(i, i + 1), objective, optimizer);
        }
    }

}
