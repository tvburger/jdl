package net.tvburger.jdl.knn;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

public class NearestNeighborsPreparer implements Regime {

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        if (!(estimationFunction instanceof NearestNeighbors nearestNeighbors)) {
            throw new IllegalArgumentException("Only works on NearestNeighbors!");
        }
        nearestNeighbors.setMemory(trainingSet);
    }
}
