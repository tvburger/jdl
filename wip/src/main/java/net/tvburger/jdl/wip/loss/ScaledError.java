package net.tvburger.jdl.wip.loss;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

public class ScaledError<E extends EstimationFunction> implements LossFunction<E> {

    private final float scale;
    private final LossFunction<E> lossFunction;

    public ScaledError(float scale, LossFunction<E> lossFunction) {
        this.scale = scale;
        this.lossFunction = lossFunction;
    }

    public float getScale() {
        return scale;
    }

    @Override
    public float calculateLoss(float error) {
        return scale * lossFunction.calculateLoss(error);
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, E estimationFunction) {
        float[] outputErrors = lossFunction.calculateOutputErrors(dataSet, estimationFunction);
        for (int i = 0; i < outputErrors.length; i++) {
            outputErrors[i] *= scale;
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, E estimationFunction) {
        float[] gradients = lossFunction.determineGradients(dataSet, estimationFunction);
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] *= scale;
        }
        return gradients;
    }
}
