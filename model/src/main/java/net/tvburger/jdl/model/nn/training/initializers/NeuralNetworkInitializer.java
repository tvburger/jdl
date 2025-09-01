package net.tvburger.jdl.model.nn.training.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.NeuronVisitor;
import net.tvburger.jdl.model.training.Initializer;

/**
 * The initializer is used to set the initial parameter value of a neuron
 */
@Strategy(Strategy.Role.INTERFACE)
public interface NeuralNetworkInitializer extends NeuronVisitor, Initializer<NeuralNetwork> {

    /**
     * {@inheritDoc}
     * <p>
     * Ensures we do not visit the layer 0 (input nodes).
     */
    @Override
    default void visitNeuron(NeuralNetwork neuralNetwork, Neuron neuron, int layerIndex, int neuronIndex) {
        if (layerIndex != 0) {
            initialize(neuralNetwork, neuron);
        }
    }

    /**
     * Initializes the parameters of the given neuron
     *
     * @param neuralNetwork the neural network the neuron belongs to
     * @param neuron        the neuron for which the parameters should be initialized
     */
    default void initialize(NeuralNetwork neuralNetwork, Neuron neuron) {
        initialize(neuron);
    }

    /**
     * Initializes the parameters of the given neuron
     *
     * @param neuron the neuron for which the parameters should be initialized
     */
    default void initialize(Neuron neuron) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    default void initialize(NeuralNetwork neuralNetwork) {
        neuralNetwork.accept(this);
    }
}
