package net.tvburger.dlp.nn;

import java.util.ArrayList;
import java.util.List;

public class MultiLayerPerceptron implements NeuralNetwork {

    private final List<List<Neuron>> layers;

    public static MultiLayerPerceptron create(int... depth) {
        List<List<Neuron>> layers = new ArrayList<>();
        for (int d = 0; d < depth.length; d++) {
            if (d == 0) {
                List<Neuron> inputNeurons = new ArrayList<>();
                for (int i = 0; i < depth[d]; i++) {
                    inputNeurons.add(new InputNeuron("Input(" + i + ")"));
                }
                layers.add(inputNeurons);
            } else {
                List<Neuron> layerNeurons = new ArrayList<>();
                List<Neuron> previousLayer = layers.getLast();
                for (int i = 0; i < depth[d]; i++) {
                    String name = d + 1 == depth.length ? "Output" : "Hidden";
                    layerNeurons.add(new Neuron(name + "(" + layers.size() + "," + i + ")", previousLayer));
                }
                layers.add(layerNeurons);
            }
        }
        return new MultiLayerPerceptron(layers);
    }

    private MultiLayerPerceptron(List<List<Neuron>> layers) {
        this.layers = layers;
    }

    @Override
    public Architecture getArchitecture() {
        return null;
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
                for (float weight : neuron.weights) {
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
            layer.forEach(Neuron::reset);
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
        System.out.println("=[ MLP Nodes Dump ]=");
        for (List<Neuron> layer : layers) {
            layer.forEach(n -> System.out.println(n.name + "{" + n.activated + ", " + n.logit + ", " + n.output + "}"));
        }
        System.out.println("====================");
    }
}
