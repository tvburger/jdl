package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.TrainableFunction;

public class OneShotRegime implements Regime {

    @Override
    public <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer, int step) {
        if (step != 1) {
            throw new IllegalStateException("Steps must be 1 but was " + step);
        }
        optimizer.optimize(estimationFunction, trainingSet, objective, 1);
    }

}
