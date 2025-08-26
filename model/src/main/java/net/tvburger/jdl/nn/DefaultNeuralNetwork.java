package net.tvburger.jdl.nn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DefaultNeuralNetwork implements NeuralNetwork {

    protected final List<List<Neuron>> layers;

    protected DefaultNeuralNetwork(List<List<Neuron>> layers) {
        this.layers = layers;
    }

    @Override
    public Architecture getArchitecture() {
        return null;
    }

    @Override
    public int getWidth(int i) {
        return layers.get(i).size();
    }

    @Override
    public int getDepth() {
        return layers.size() - 1;
    }

    @Override
    public Neuron getNeuron(int layer, int index) {
        return layers.get(layer).get(index);
    }

    @Override
    public Map<Neuron, Float> getOutputConnections(int layer, int index) {
        Map<Neuron, Float> connections = new HashMap<>();
        Neuron source = layers.get(layer).get(index);
        if (layer < layers.size() - 1) {
            for (Neuron target : layers.get(layer + 1)) {
                Float weight = target.findWeight(source);
                if (weight != null) {
                    connections.put(target, weight);
                }
            }
        }
        return connections;
    }

    @Override
    public int getParameterCount() {
        int parameterCount = 0;
        for (List<Neuron> layer : layers) {
            for (Neuron neuron : layer) {
                if (neuron instanceof InputNeuron) {
                    continue;
                }
                parameterCount += neuron.getWeights().length + 1;
            }
        }
        return parameterCount;
    }

    @Override
    public float[] getParameters() {
        float[] parameters = new float[getParameterCount()];
        int i = 0;
        for (List<Neuron> layer : layers) {
            for (Neuron neuron : layer) {
                if (neuron instanceof InputNeuron) {
                    continue;
                }
                for (float weight : neuron.getWeights()) {
                    parameters[i++] = weight;
                }
                parameters[i++] = neuron.getBias();
            }
        }
        return parameters;
    }

    @Override
    public float[] estimate(float... inputs) {
        for (List<Neuron> layer : layers) {
            layer.forEach(Neuron::deactivate);
        }
        for (int i = 0; i < inputs.length; i++) {
            ((InputNeuron) layers.get(0).get(i)).setInputValue(inputs[i]);
        }
        for (List<Neuron> layer : layers) {
            layer.forEach(Neuron::activate);
        }
        List<Neuron> outputLayer = layers.get(layers.size() - 1);
        float[] outputs = new float[outputLayer.size()];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = outputLayer.get(i).getOutput();
        }
        return outputs;
    }

    @Override
    public void init(Initializer initializer) {
        for (List<Neuron> layer : layers) {
            layer.forEach(initializer::initialize);
        }
    }

    @Override
    public void dumpNodeOutputs() {
        System.out.println("=[ Neural Network Node Dump ]=");
        for (List<Neuron> layer : layers) {
            layer.forEach(n -> System.out.println(n.toString()));
        }
        System.out.println("====================");
    }

    @Override
    public ActivationFunction getOutputActivationFunction() {
        return layers.get(layers.size() - 1).getFirst().getActivationFunction();
    }
}
