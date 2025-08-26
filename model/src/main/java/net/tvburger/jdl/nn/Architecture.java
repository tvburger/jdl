package net.tvburger.jdl.nn;

import net.tvburger.jdl.learning.LossFunction;

public interface Architecture {

    int getParameterCount();

    float[] getParameters();

    ActivationFunction getActivationFunction();

    LossFunction getLossFunction();

}
