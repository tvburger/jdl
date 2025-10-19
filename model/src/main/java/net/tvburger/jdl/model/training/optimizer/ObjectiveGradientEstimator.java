package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.TrainableFunction;

public class ObjectiveGradientEstimator<N extends Number> {

//    public Vector<N> estimateGradients(DataSet<N> trainingSet, TrainableFunction<N> trainableFunction, ObjectiveFunction<N> objectiveFunction) {
//        if (trainingSet == null || trainingSet.isEmpty()) {
//            throw new IllegalArgumentException("trainingSet must not be empty");
//        }
//        return determineMeanGradient(trainingSet, trainableFunction, objectiveFunction);
//    }
//
//    private Vector<N> determineMeanGradient(DataSet<N> trainingSet, TrainableFunction<N> trainableFunction, ObjectiveFunction<N> objectiveFunction) {
//        Pair<Vector<N>, Integer> accumulatedGradients = accumulateGradients(trainingSet, trainableFunction, objectiveFunction);
//        int total = accumulatedGradients.right();
//        Vector<N> accumulatedGradient = accumulatedGradients.left();
//
//        Vector<N> averageGradient;
//        if (total == 1) {
//            averageGradient = accumulatedGradient;
//        } else if (accumulatedGradient == null) {
//            averageGradient = null;
//        } else {
//            JavaNumberTypeSupport<N> typeSupport = trainableFunction.getCurrentNumberType();
//            averageGradient = accumulatedGradient.multiply(typeSupport.inverse(typeSupport.valueOf(total)));
//        }
//        return averageGradient;
//    }
//
//    private Pair<Vector<N>, Integer> accumulateGradients(DataSet<N> trainingSet, TrainableFunction<N> trainableFunction, ObjectiveFunction<N> objectiveFunction) {
//        Vector<N> accumulatedGradient = null;
//        int total = 0;
//        for (DataSet.Sample<N> sample : trainingSet) {
//            if (accumulatedGradient == null) {
//                accumulatedGradient = determineGradient(sample, trainableFunction, objectiveFunction);
//            } else {
//                accumulatedGradient = accumulatedGradient.add(determineGradient(sample, trainableFunction, objectiveFunction));
//            }
//            total++;
//        }
//        return Pair.of(accumulatedGradient, total);
//    }

    public Vector<N> determineGradient(DataSet.Sample<N> sample, TrainableFunction<N> estimationFunction, ObjectiveFunction<N> objectiveFunction) {
        N[] estimated = estimationFunction.estimate(sample.features());
        N[] target = sample.targetOutputs();
        N[] gradients = objectiveFunction.calculateGradient_dJ_da(1, estimated, target);
        return Vectors.of(estimationFunction.getCurrentNumberType(), gradients).transpose();
    }

}
