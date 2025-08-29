package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.Mediator;
import net.tvburger.jdl.common.utils.Pair;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

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
    private final Map<Neuron, Pair<Integer, Integer>> positions;
    private final Map<Neuron, List<Pair<Neuron, Float>>> connections;

    /**
     * Constructs a neural network for the provided layers.
     *
     * @param layers the layers of the neural network
     */
    public DefaultNeuralNetwork(List<List<? extends Neuron>> layers) {
        this.layers = layers;
        this.positions = NeuralNetworks.getNeuronPositions(layers);
        this.connections = NeuralNetworks.getNeuronConnections(layers);
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
        Neuron neuron = getNeuron(layer, index);
        List<Pair<Neuron, Float>> pairs = connections.get(neuron);
        Map<Neuron, Float> outputConnections = new IdentityHashMap<>();
        if (pairs != null) {
            for (Pair<Neuron, Float> pair : pairs) {
                outputConnections.put(pair.left(), pair.right());
            }
        }
        return outputConnections;
    }

    @Override
    public Set<Neuron> getTargetNeurons(Neuron neuron) {
        List<Pair<Neuron, Float>> pairs = connections.get(neuron);
        return pairs == null ? Set.of() : pairs.stream().map(Pair::left).collect(Collectors.toSet());
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
     * Accepts a visitor to inspect the complete neural network
     *
     * @param visitor the visitor to apply to this network
     */
    @Override
    public void accept(NeuronVisitor visitor) {
        if (visitor == null) {
            return;
        }
        visitor.enterNetwork(this);

        int depth = getDepth();
        for (int layer = 1; layer <= depth; layer++) {
            visitor.enterLayer(this, layer);

            int width = getWidth(layer);
            for (int j = 0; j < width; j++) {
                Neuron neuron = getNeuron(layer, j, Neuron.class);
                visitor.visitNeuron(this, neuron, layer, j);

                Map<Neuron, Float> outs = getOutputConnections(layer, j);
                if (outs != null) {
                    for (Map.Entry<Neuron, Float> connection : outs.entrySet()) {
                        Neuron target = connection.getKey();
                        Pair<Integer, Integer> targetPosition = positions.get(target);
                        visitor.visitConnection(this, neuron, layer, j,
                                target, targetPosition.left(), targetPosition.right(), connection.getValue());
                    }
                }
            }

            visitor.exitLayer(this, layer);
        }

        visitor.exitNetwork(this);
    }
}
