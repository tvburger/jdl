package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.Visitor;

/**
 * A visitor interface for traversing a {@link NeuralNetwork}.
 *
 * <p>This interface follows the <em>Visitor Pattern</em>, allowing clients to
 * perform operations on neurons, layers, and connections without modifying the
 * network structure. Implementations can override any of the default methods
 * to handle specific events during traversal.
 *
 * <p>All methods have default empty implementations, so you can override only
 * the methods you need.
 *
 * @see NeuralNetwork
 * @see Neuron
 */
@Visitor
public interface NeuronVisitor {

    /**
     * Called when entering the entire network before any layers or neurons
     * are visited.
     *
     * @param neuralNetwork the neural network being traversed
     */
    default void enterNetwork(NeuralNetwork neuralNetwork) {
    }

    /**
     * Called after all layers and neurons have been visited in the network.
     *
     * @param neuralNetwork the neural network that was traversed
     */
    default void exitNetwork(NeuralNetwork neuralNetwork) {
    }

    /**
     * Called when entering a specific layer in the network.
     *
     * @param neuralNetwork the neural network being traversed
     * @param layerIndex    the index of the layer being entered
     */
    default void enterLayer(NeuralNetwork neuralNetwork, int layerIndex) {
    }

    /**
     * Called after all neurons in a specific layer have been visited.
     *
     * @param neuralNetwork the neural network being traversed
     * @param layerIndex    the index of the layer being exited
     */
    default void exitLayer(NeuralNetwork neuralNetwork, int layerIndex) {
    }

    /**
     * Called when visiting a single neuron in a specific layer.
     *
     * @param neuralNetwork the neural network being traversed
     * @param neuron        the neuron being visited
     * @param layerIndex    the index of the layer containing the neuron
     * @param neuronIndex   the index of the neuron within its layer
     */
    default void visitNeuron(NeuralNetwork neuralNetwork, Neuron neuron, int layerIndex, int neuronIndex) {
    }

    /**
     * Called when visiting a connection (synapse) between two neurons.
     *
     * @param neuralNetwork the neural network being traversed
     * @param source        the source neuron of the connection
     * @param fromLayer     the layer index of the source neuron
     * @param fromIndex     the index of the source neuron within its layer
     * @param target        the target neuron of the connection
     * @param toLayer       the layer index of the target neuron
     * @param toIndex       the index of the target neuron within its layer
     * @param weight        the weight of the connection
     */
    default void visitConnection(NeuralNetwork neuralNetwork, Neuron source, int fromLayer, int fromIndex,
                                 Neuron target, int toLayer, int toIndex, float weight) {
    }

}
