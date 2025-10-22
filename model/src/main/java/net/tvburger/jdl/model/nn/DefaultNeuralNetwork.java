package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Mediator;
import net.tvburger.jdl.common.utils.Pair;

import java.util.*;

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
    private final Map<Neuron, List<Neuron>> connections;

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
    public int getWidth(int l) {
        return layers.get(l).size();
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
    public Neuron getNeuron(int l, int j) {
        return layers.get(l).get(j);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <N extends Neuron> N getNeuron(int l, int j, Class<N> classType) {
        return classType.cast(getNeuron(l, j));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Map<Neuron, Float> getOutputConnections(int l, int j) {
        Neuron neuron = getNeuron(l, j);
        List<Neuron> connectedNeurons = connections.get(neuron);
        Map<Neuron, Float> outputConnections = new IdentityHashMap<>();
        if (connectedNeurons != null) {
            for (Neuron connectedNeuron : connectedNeurons) {
                outputConnections.put(connectedNeuron, connectedNeuron.getWeight(neuron));
            }
        }
        return outputConnections;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Set<Neuron> getTargetNeurons(Neuron neuron) {
        List<Neuron> neurons = connections.get(neuron);
        return neurons == null ? Set.of() : new HashSet<>(neurons);
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
                parameterCount += neuron.getParameterCount();
            }
        }
        return parameterCount;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array<Float> getParameters() {
        Array<Float> parameters = Array.of(new Float[getParameterCount()]);
        int i = 0;
        for (List<? extends Neuron> layer : layers) {
            for (Neuron neuron : layer) {
                if (neuron instanceof InputNeuron) {
                    continue;
                }
                Array<Float> neuronParameters = neuron.getParameters();
                parameters.set(neuronParameters, i, neuronParameters.length());
            }
        }
        return parameters;
    }

    @Override
    public Float getParameter(int p) {
        return getParameters().get(p);
    }

    @Override
    public void setParameters(Array<Float> values) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setParameter(int p, Float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public JavaNumberTypeSupport<Float> getNumberTypeSupport() {
        return JavaNumberTypeSupport.FLOAT;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array<Float> estimate(Array<Float> inputs) {
        for (List<? extends Neuron> layer : layers) {
            layer.forEach(Neuron::deactivate);
        }
        for (int j = 0; j < inputs.length(); j++) {
            ((InputNeuron) layers.getFirst().get(j)).setInputValue(inputs.get(j));
        }
        for (List<? extends Neuron> layer : layers) {
            layer.forEach(Neuron::activate);
        }
        List<? extends Neuron> outputLayer = layers.getLast();
        Array<Float> outputs = Array.of(new Float[outputLayer.size()]);
        for (int j = 0; j < outputs.length(); j++) {
            outputs.set(j, outputLayer.get(j).getOutput());
        }
        return outputs;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int arity() {
        return layers.getFirst().size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int coArity() {
        return layers.getLast().size();
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
        for (int l = 1; l <= depth; l++) {
            visitor.enterLayer(this, l);

            int width = getWidth(l);
            for (int j = 0; j < width; j++) {
                Neuron neuron = getNeuron(l, j, Neuron.class);
                visitor.visitNeuron(this, neuron, l, j);

                Map<Neuron, Float> outs = getOutputConnections(l, j);
                if (outs != null) {
                    for (Map.Entry<Neuron, Float> connection : outs.entrySet()) {
                        Neuron target = connection.getKey();
                        Pair<Integer, Integer> targetPosition = positions.get(target);
                        visitor.visitConnection(this, neuron, l, j,
                                target, targetPosition.left(), targetPosition.right(), connection.getValue());
                    }
                }
            }

            visitor.exitLayer(this, l);
        }

        visitor.exitNetwork(this);
    }
}
