package net.tvburger.jdl.mlp;

import net.tvburger.jdl.nn.DefaultNeuralNetwork;
import net.tvburger.jdl.nn.ActivationFunction;
import net.tvburger.jdl.nn.InputNeuron;
import net.tvburger.jdl.nn.Neuron;
import net.tvburger.jdl.nn.activations.Activations;

import java.util.ArrayList;
import java.util.List;

public class MultiLayerPerceptron extends DefaultNeuralNetwork {

    public static MultiLayerPerceptron create(ActivationFunction outputActivationFunction, int... depth) {
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
                    String name;
                    ActivationFunction activationFunction;
                    if (depth.length == d + 1) {
                        name = "Output";
                        activationFunction = outputActivationFunction;
                    } else {
                        name = "Hidden";
                        activationFunction = Activations.reLU();
                    }
                    layerNeurons.add(new Neuron(name + "(" + layers.size() + "," + i + ")", previousLayer, activationFunction));
                }
                layers.add(layerNeurons);
            }
        }
        return new MultiLayerPerceptron(layers);
    }

    private MultiLayerPerceptron(List<List<Neuron>> layers) {
        super(layers);
    }

}
