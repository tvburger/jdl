package net.tvburger.jdl.model.loss;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

@Strategy(role = Strategy.Role.CONCRETE)
public class SquaredError implements LossFunction {

    @Override
    public float calculateLoss(DataSet dataSet, EstimationFunction estimationFunction) {
        float loss = 0.0f;
        for (DataSet.Sample sample : dataSet) {
            float[] estimate = estimationFunction.estimate(sample.features());
            for (int i = 0; i < sample.targetCount(); i++) {
                float error = sample.features()[i] - estimate[i];
                loss += error * error;
            }
        }
        return loss;
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, EstimationFunction estimationFunction) {
        float[] outputErrors = new float[estimationFunction.coArity()];
        for (DataSet.Sample sample : dataSet) {
            float[] predictedOutputs = estimationFunction.estimate(sample.features());
            for (int i = 0; i < predictedOutputs.length; i++) {
                outputErrors[i] += predictedOutputs[i] - sample.targetOutputs()[i];
            }
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, EstimationFunction estimationFunction) {
        float[] gradients = calculateOutputErrors(dataSet, estimationFunction);
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] *= 2.0f;
        }
        return gradients;
    }

}
