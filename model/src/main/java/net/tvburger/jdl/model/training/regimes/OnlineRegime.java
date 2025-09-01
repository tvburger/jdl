package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

@Strategy(Strategy.Role.CONCRETE)
public final class OnlineRegime implements Regime {

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        for (int i = 0; i < trainingSet.samples().size(); i++) {
            optimizer.optimize(estimationFunction, trainingSet, objective);
        }
    }

}
