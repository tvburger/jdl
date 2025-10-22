package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.model.nn.*;
import net.tvburger.jdl.model.scalars.activations.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class FullyConnectedLayer implements ConvolutionalNetworkLayer {

    private final ConvolutionalShape inputShape;
    private final ConvolutionalShape outputShape;
    private final NeuralNetwork neuralNetwork;

    public static FullyConnectedLayer create(ConvolutionalShape inputShape, int dimensions, ActivationFunction activationFunction) {
        ConvolutionalShape outputShape = new ConvolutionalShape(dimensions, 1);
        List<InputNeuron> inputNodes = new ArrayList<>();
        for (int i = 0; i < inputShape.getElements().length(); i++) {
            inputNodes.add(new InputNeuron("Input(" + i + ")"));
        }
        List<Neuron> outputNodes = new ArrayList<>();
        for (int i = 0; i < dimensions; i++) {
            outputNodes.add(ActivationsCachedNeuron.create("Neuron(" + i + ")", inputNodes, activationFunction));
        }
        return new FullyConnectedLayer(inputShape, outputShape, new DefaultNeuralNetwork(List.of(inputNodes, outputNodes)));
    }

    public FullyConnectedLayer(ConvolutionalShape inputShape, ConvolutionalShape outputShape, NeuralNetwork neuralNetwork) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.neuralNetwork = neuralNetwork;
    }

    @Override
    public ConvolutionalShape getInputShape() {
        return inputShape;
    }

    @Override
    public ConvolutionalShape getOutputShape() {
        return outputShape;
    }

    @Override
    public ConvolutionalShape transform(ConvolutionalShape input) {
        Array<Float> estimate = neuralNetwork.estimate(input.getElements());
        return outputShape.withElements(estimate);
    }

}