package net.tvburger.jdl.model.loss;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

@Decorator
public class MeanError implements LossFunction {

    private final LossFunction lossFunction;

    public MeanError(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    @Override
    public float calculateLoss(DataSet dataSet, EstimationFunction estimationFunction) {
        return lossFunction.calculateLoss(dataSet, estimationFunction) / dataSet.samples().size();
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, EstimationFunction estimationFunction) {
        float[] outputErrors = lossFunction.calculateOutputErrors(dataSet, estimationFunction);
        for (int i = 0; i < outputErrors.length; i++) {
            outputErrors[i] /= dataSet.samples().size();
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, EstimationFunction estimationFunction) {
        float[] gradients = lossFunction.determineGradients(dataSet, estimationFunction);
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= dataSet.samples().size();
        }
        return gradients;
    }

}
