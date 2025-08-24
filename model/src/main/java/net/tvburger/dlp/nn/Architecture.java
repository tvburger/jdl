package net.tvburger.dlp.nn;

import net.tvburger.dlp.ActivationFunction;
import net.tvburger.dlp.LossFunction;

public interface Architecture {

    int getParameterCount();

    float[] getParameters();

    ActivationFunction getActivationFunction();

    LossFunction getLossFunction();

}
