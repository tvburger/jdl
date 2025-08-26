package net.tvburger.jdl.learning;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.nn.NeuralNetwork;

public interface LossFunction {

    float calculateLoss(float error);

    default float calculateLoss(DataSet dataSet, NeuralNetwork neuralNetwork) {
        float[] outputErrors = calculateOutputErrors(dataSet, neuralNetwork);
        float totalLoss = 0.0f;
        for (float outputError : outputErrors) {
            totalLoss += calculateLoss(outputError);
        }
        return totalLoss / outputErrors.length;
    }

    default float calculateAggregateError(DataSet dataSet, NeuralNetwork neuralNetwork) {
        float[] outputErrors = calculateOutputErrors(dataSet, neuralNetwork);
        float totalError = 0.0f;
        for (float outputError : outputErrors) {
            totalError += outputError;
        }
        return totalError / outputErrors.length;
    }

    float[] calculateOutputErrors(DataSet dataSet, NeuralNetwork neuralNetwork);

    float[] determineGradients(DataSet dataSet, NeuralNetwork neuralNetwork);

}
