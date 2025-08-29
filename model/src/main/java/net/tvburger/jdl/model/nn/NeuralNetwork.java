package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.nn.initializers.NeuralNetworkInitializer;

import java.util.Map;
import java.util.Set;

/**
 * Represents a neural network as a specific kind of {@link EstimationFunction}.
 * <p>
 * A neural network consists of multiple layers of {@link Neuron}s connected
 * through weighted edges. This interface exposes structural information
 * (layers, neurons, connections), parameter management, and initialization
 * capabilities. It does not prescribe a specific network architecture
 * (e.g., fully connected, convolutional, recurrent).
 * </p>
 *
 * <h2>Responsibilities:</h2>
 * <ul>
 *   <li>Expose the <b>topology</b> of the network in terms of layers and neurons.</li>
 *   <li>Provide access to <b>parameters</b> (weights, biases) for training
 *       and persistence.</li>
 *   <li>Support <b>initialization</b> of parameters through an {@link NeuralNetworkInitializer}.</li>
 *   <li>Inherit evaluation capabilities from {@link EstimationFunction},
 *       such as mapping inputs to outputs.</li>
 * </ul>
 */
@DomainObject
public interface NeuralNetwork extends EstimationFunction {

    /**
     * Returns the number of neurons in the given layer.
     *
     * @param layer the layer index (0-based, where 0 is the input layer)
     * @return the number of neurons in that layer
     * @throws IndexOutOfBoundsException if the layer index is invalid
     */
    int getWidth(int layer);

    /**
     * Returns the total number of layers in the network, including input and output layers.
     *
     * @return the depth of the network
     */
    int getDepth();

    /**
     * Returns a specific neuron in the network by its layer and index.
     *
     * @param layer the layer index (0-based)
     * @param index the index of the neuron within the layer (0-based)
     * @return the neuron at the specified position
     * @throws IndexOutOfBoundsException if the indices are invalid
     */
    Neuron getNeuron(int layer, int index);

    /**
     * Returns a specific neuron in the network by its layer and index.
     *
     * @param layer     the layer index (0-based)
     * @param index     the index of the neuron within the layer (0-based)
     * @param classType the class of the neuron to be returned
     * @return the neuron at the specified position
     * @throws IndexOutOfBoundsException if the indices are invalid
     */
    <N extends Neuron> N getNeuron(int layer, int index, Class<N> classType);

    /**
     * Returns the outgoing connections of a given neuron in the network.
     * <p>
     * Each connection is represented as a mapping from a target neuron
     * to the weight of the connection.
     * </p>
     *
     * @param layer the layer index of the source neuron
     * @param index the index of the source neuron within the layer
     * @return a map of target neurons to their associated connection weights
     */
    Map<Neuron, Float> getOutputConnections(int layer, int index);

    /**
     * Returns the target neurons of the given neuron (thus the downstream nodeds).
     *
     * @param neuron the neuron for which the targets are returned
     * @return the target neurons
     */
    Set<Neuron> getTargetNeurons(Neuron neuron);

    /**
     * Returns the total number of trainable parameters in the network
     * (typically weights + biases).
     *
     * @return the number of parameters
     */
    int getParameterCount();

    /**
     * Returns all parameters (weights, biases) of the network as a flat array.
     * <p>
     * The order of parameters is implementation-defined but must be consistent
     * across calls to enable saving and restoring network state.
     * </p>
     *
     * @return a flat array of parameters
     */
    float[] getParameters();

    /**
     * Accepts a {@link net.tvburger.jdl.model.nn.NeuronVisitor} to traverse
     * the structure of this neural network.
     * This allows external visitors to analyze, transform, or export the
     * network structure without coupling to its internal implementation.
     *
     * @param visitor the visitor to apply to this network
     */
    void accept(NeuronVisitor visitor);

}
