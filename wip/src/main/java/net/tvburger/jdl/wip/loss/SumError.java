package net.tvburger.jdl.wip.loss;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

public class SumError<E extends EstimationFunction> implements LossFunction<E> {

    private final LossFunction<E> lossFunction;

    public SumError(LossFunction<E> lossFunction) {
        this.lossFunction = lossFunction;
    }

    @Override
    public float calculateLoss(float error) {
        return lossFunction.calculateLoss(error);
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, E estimationFunction) {
        float[] totalOutputErrors = new float[estimationFunction.coArity()];
        dataSet.forEach(s -> {
            float[] predictedOutputs = estimationFunction.estimate(s.features());
            for (int i = 0; i < predictedOutputs.length; i++) {
                totalOutputErrors[i] += predictedOutputs[i] - s.targetOutputs()[i];
            }
        });
        return totalOutputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, E estimationFunction) {
        float[] totalGradients = new float[estimationFunction.coArity()];
        dataSet.forEach(s -> {
            float[] gradients = lossFunction.determineGradients(dataSet, estimationFunction);
            for (int i = 0; i < gradients.length; i++) {
                totalGradients[i] += gradients[i] - s.targetOutputs()[i];
            }
        });
        return totalGradients;
    }

}
