package net.tvburger.dlp.nn;

import net.tvburger.dlp.learning.LossFunction;

public interface Architecture {

    int getParameterCount();

    float[] getParameters();

    ActivationFunction getActivationFunction();

    LossFunction getLossFunction();

}
