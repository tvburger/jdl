package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.TrainableFunction;

public class ObjectiveGradientEstimator<N extends Number> {

    public Vector<N> determineGradient(DataSet.Sample<N> sample, TrainableFunction<N> estimationFunction, ObjectiveFunction<N> objectiveFunction) {
        Array<N> estimated = estimationFunction.estimate(sample.features());
        Array<N> target = sample.targetOutputs();
        Array<N> gradients = objectiveFunction.calculateGradient_dJ_da(1, estimated, target);
        return new TypedVector<>(gradients, true, estimationFunction.getNumberTypeSupport());
    }

}
