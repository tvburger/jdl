package net.tvburger.jdl.model.nn;

public interface NeuronVisitor {

    default void enterNetwork(NeuralNetwork neuralNetwork) {
    }

    default void exitNetwork(NeuralNetwork neuralNetwork) {
    }

    default void enterLayer(NeuralNetwork neuralNetwork, int layerIndex) {
    }

    default void exitLayer(NeuralNetwork neuralNetwork, int layerIndex) {
    }

    default void visitNeuron(NeuralNetwork neuralNetwork, Neuron neuron, int layerIndex, int neuronIndex) {
    }

    default void visitConnection(NeuralNetwork neuralNetwork, Neuron source, int fromLayer, int fromIndex,
                                 Neuron target, int toLayer, int toIndex, float weight) {
    }

}
