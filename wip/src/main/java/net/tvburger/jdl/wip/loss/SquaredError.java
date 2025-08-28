package net.tvburger.jdl.wip.loss;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

public class SquaredError<E extends EstimationFunction> implements LossFunction<E> {

    @Override
    public float calculateLoss(float error) {
        return error * error;
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, E estimationFunction) {
        float[] outputErrors = new float[estimationFunction.coArity()];
        int totalSamples = dataSet.samples().size();
        for (DataSet.Sample sample : dataSet.samples()) {
            float[] predictedOutputs = estimationFunction.estimate(sample.features());
            for (int i = 0; i < predictedOutputs.length; i++) {
                outputErrors[i] += (predictedOutputs[i] - sample.targetOutputs()[i]) / totalSamples;
            }
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, E estimationFunction) {
        float[] gradients = calculateOutputErrors(dataSet, estimationFunction);
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] *= 2.0f;
        }
        return gradients;
    }

}
