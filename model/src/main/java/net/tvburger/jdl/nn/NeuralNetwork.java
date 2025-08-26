package net.tvburger.jdl.nn;

import java.util.Map;

public interface NeuralNetwork {

    Architecture getArchitecture();

    int getWidth(int layer);

    int getDepth();

    Neuron getNeuron(int layer, int index);

    Map<Neuron, Float> getOutputConnections(int layer, int index);

    int getParameterCount();

    float[] getParameters();

    float[] estimate(float[] inputs);

    void init(Initializer initializer);

    void dumpNodeOutputs();

    ActivationFunction getOutputActivationFunction();

}
