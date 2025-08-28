package net.tvburger.jdl.wip.loss;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

public class MeanError<E extends EstimationFunction> implements LossFunction<E> {

    private final LossFunction<E> lossFunction;

    public MeanError(LossFunction<E> lossFunction) {
        this.lossFunction = lossFunction;
    }

    @Override
    public float calculateLoss(float error) {
        return lossFunction.calculateLoss(error);
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, E estimationFunction) {
        float[] outputErrors = new float[estimationFunction.coArity()];
        int totalSamples = dataSet.samples().size();
        for (DataSet.Sample sample : dataSet) {
            float[] predictedOutputs = estimationFunction.estimate(sample.features());
            for (int i = 0; i < predictedOutputs.length; i++) {
                outputErrors[i] += (predictedOutputs[i] - sample.targetOutputs()[i]) / totalSamples;
            }
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, E neuralNetwork) {
        return lossFunction.determineGradients(dataSet, neuralNetwork);
    }

}
