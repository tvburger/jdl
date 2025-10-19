package net.tvburger.jdl.knn;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.TrainableFunction;

public class NearestNeighborsPreparer implements Regime {

    @SuppressWarnings("unchecked")
    @Override
    public <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer, int step) {
        if (!(estimationFunction instanceof NearestNeighbors nearestNeighbors)) {
            throw new IllegalArgumentException("Only works on NearestNeighbors!");
        }
        nearestNeighbors.setMemory((DataSet<Float>) trainingSet);
    }
}
