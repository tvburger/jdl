package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.Mediator;
import net.tvburger.jdl.model.nn.initializers.Initializer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Provides a standard implementation for a neural network.
 * <p>
 * A neural network is modeled here as an extensible estimation strategy: it accepts
 * inputs (features), transforms them through a series of layers composed of {@link Neuron}s
 * and weighted connections, and produces outputs (predictions).
 * </p>
 */
@Mediator
public class DefaultNeuralNetwork implements NeuralNetwork {

    private final List<List<? extends Neuron>> layers;

    /**
     * Constructs a neural network for the provided layers.
     *
     * @param layers the layers of the neural network
     */
    public DefaultNeuralNetwork(List<List<? extends Neuron>> layers) {
        this.layers = layers;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getWidth(int i) {
        return layers.get(i).size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getDepth() {
        return layers.size() - 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Neuron getNeuron(int layer, int index) {
        return layers.get(layer).get(index);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <N extends Neuron> N getNeuron(int layer, int index, Class<N> classType) {
        return classType.cast(getNeuron(layer, index));
    }

    /**
     * {@inheritDoc}
     */
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

    /**
     * {@inheritDoc}
     */
    @Override
    public int getParameterCount() {
        int parameterCount = 0;
        for (List<? extends Neuron> layer : layers) {
            for (Neuron neuron : layer) {
                if (neuron instanceof InputNeuron) {
                    continue;
                }
                parameterCount += neuron.getWeights().length + 1;
            }
        }
        return parameterCount;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] getParameters() {
        float[] parameters = new float[getParameterCount()];
        int i = 0;
        for (List<? extends Neuron> layer : layers) {
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

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] estimate(float... inputs) {
        for (List<? extends Neuron> layer : layers) {
            layer.forEach(Neuron::deactivate);
        }
        for (int i = 0; i < inputs.length; i++) {
            ((InputNeuron) layers.get(0).get(i)).setInputValue(inputs[i]);
        }
        for (List<? extends Neuron> layer : layers) {
            layer.forEach(Neuron::activate);
        }
        List<? extends Neuron> outputLayer = layers.get(layers.size() - 1);
        float[] outputs = new float[outputLayer.size()];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = outputLayer.get(i).getOutput();
        }
        return outputs;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int arity() {
        return layers.get(0).size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int coArity() {
        return layers.get(layers.size() - 1).size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void init(Initializer initializer) {
        for (List<? extends Neuron> layer : layers) {
            layer.forEach(initializer::initialize);
        }
    }

}
