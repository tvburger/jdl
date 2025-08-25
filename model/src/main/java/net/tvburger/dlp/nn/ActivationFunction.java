package net.tvburger.dlp.nn;

public interface ActivationFunction {

    String name();

    float activate(float logit);

}
