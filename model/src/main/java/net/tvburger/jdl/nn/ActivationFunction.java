package net.tvburger.jdl.nn;

public interface ActivationFunction {

    String name();

    float activate(float logit);

}
