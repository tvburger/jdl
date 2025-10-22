package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.nn.DefaultNeuralNetwork;
import net.tvburger.jdl.model.nn.InputNeuron;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.ArrayList;
import java.util.List;

public class Adaline extends DefaultNeuralNetwork {

    public static Adaline create(int inputs, int outputs) {
        List<InputNeuron> inputNodes = new ArrayList<>();
        for (int i = 0; i < inputs; i++) {
            inputNodes.add(new InputNeuron("Input(" + i + ")"));
        }
        List<Neuron> outputNodes = new ArrayList<>();
        for (int i = 0; i < outputs; i++) {
            outputNodes.add(Neuron.create("Adaline(" + i + ")", inputNodes));
        }
        return new Adaline(List.of(inputNodes, outputNodes));
    }

    private Adaline(List<List<? extends Neuron>> layers) {
        super(layers);
    }

    public Array<Boolean> classify(Array<Float> inputs) {
        Array<Float> estimate = estimate(inputs);
        Array<Boolean> classifications = Array.of(new Boolean[estimate.length()]);
        for (int i = 0; i < estimate.length(); i++) {
            classifications.set(i, Floats.greaterThan(estimate.get(i), 0.0f));
        }
        return classifications;
    }

}
