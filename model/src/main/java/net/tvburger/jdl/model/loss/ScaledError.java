package net.tvburger.jdl.model.loss;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

@Decorator
public class ScaledError implements LossFunction {

    private final float scale;
    private final LossFunction lossFunction;

    public ScaledError(float scale, LossFunction lossFunction) {
        this.scale = scale;
        this.lossFunction = lossFunction;
    }

    public float getScale() {
        return scale;
    }

    @Override
    public float calculateLoss(DataSet dataSet, EstimationFunction estimationFunction) {
        return lossFunction.calculateLoss(dataSet, estimationFunction) * scale;
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, EstimationFunction estimationFunction) {
        float[] outputErrors = lossFunction.calculateOutputErrors(dataSet, estimationFunction);
        for (int i = 0; i < outputErrors.length; i++) {
            outputErrors[i] *= scale;
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, EstimationFunction estimationFunction) {
        float[] gradients = lossFunction.determineGradients(dataSet, estimationFunction);
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] *= scale;
        }
        return gradients;
    }

}
