package net.tvburger.dlp.nn;

public interface NeuralNetwork {

    Architecture getArchitecture();

    int getParameterCount();

    float[] getParameters();

    float[] estimate(float[] inputs);

    void init(Initializer initializer);

    void dumpNodeOutputs();

}
