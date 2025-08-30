package net.tvburger.jdl.mlp;

import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.initializers.NeuralNetworkInitializer;

public class XorInitializer implements NeuralNetworkInitializer {

    @Override
    public void visitNeuron(NeuralNetwork neuralNetwork, Neuron neuron, int layerIndex, int neuronIndex) {
        if (layerIndex == 1) {
            neuron.setWeights(new float[]{5.0f, 5.0f});
            neuron.setBias(neuronIndex == 1 ? -2.5f : -7.5f);
        } else if (layerIndex == 2) {
            neuron.setBias(-2.5f);
        }
    }

}
