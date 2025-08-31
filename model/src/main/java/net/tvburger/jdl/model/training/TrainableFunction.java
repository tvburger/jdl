package net.tvburger.jdl.model.training;

import net.tvburger.jdl.model.EstimationFunction;

public interface TrainableFunction extends EstimationFunction {

    default int getParameterCount() {
        return getParameters().length;
    }

    float[] getParameters();

    default void setParameters(float[] values) {
        float[] parameters = getParameters();
        if (values.length != parameters.length) {
            throw new IllegalArgumentException();
        }
        System.arraycopy(values, 0, parameters, 0, parameters.length);
    }

    default float getParameter(int p) {
        return getParameters()[p];
    }

    default void setParameter(int p, float value) {
        getParameters()[p] = value;
    }

}
