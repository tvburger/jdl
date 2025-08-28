package net.tvburger.jdl.wip.error;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

public interface ErrorFunction<E extends EstimationFunction> {

    float calculateError(float error);

    default float calculateAggregateError(DataSet dataSet, E estimationFunction) {
        float[] outputErrors = calculateOutputErrors(dataSet, estimationFunction);
        float totalError = 0.0f;
        for (float outputError : outputErrors) {
            totalError += outputError;
        }
        return totalError / outputErrors.length;
    }

    float[] calculateOutputErrors(DataSet dataSet, E estimationFunction);

    float[] determineGradients(DataSet dataSet, E estimationFunction);

}
