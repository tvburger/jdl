package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.StaticUtility;
import net.tvburger.jdl.common.utils.Pair;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility methods for Neural Networks
 */
@StaticUtility
public final class NeuralNetworks {

    private NeuralNetworks() {
    }

    /**
     * Dumps the neural network nodes to stdout
     *
     * @param neuralNetwork neural network to dump
     */
    public static void dump(NeuralNetwork neuralNetwork) {
        System.out.println("=[ Neural Network Node Dump ]=");
        for (int l = 0; l < neuralNetwork.getDepth() + 1; l++) {
            for (int i = 0; i < neuralNetwork.getWidth(l); i++) {
                System.out.println(neuralNetwork.getNeuron(l, i));
            }
        }
        System.out.println("====================");
    }

    /**
     * Returns a map containing the node positions in the layer
     *
     * @param layers the layers of a neural network
     * @return the node positions
     */
    public static Map<Neuron, Pair<Integer, Integer>> getNeuronPositions(List<List<? extends Neuron>> layers) {
        Map<Neuron, Pair<Integer, Integer>> positions = new IdentityHashMap<>();
        for (int i = 0; i < layers.size(); i++) {
            List<? extends Neuron> layer = layers.get(i);
            for (int j = 0; j < layer.size(); j++) {
                positions.put(layer.get(j), Pair.of(i, j));
            }
        }
        return positions;
    }

    /**
     * Returns a map containing the node connections to the next layer with their corresponding weights
     *
     * @param layers the layers of a neural network
     * @return the node connections
     */
    public static Map<Neuron, List<Pair<Neuron, Float>>> getNeuronConnections(List<List<? extends Neuron>> layers) {
        Map<Neuron, List<Pair<Neuron, Float>>> connections = new IdentityHashMap<>();
        for (List<? extends Neuron> layer : layers) {
            for (Neuron neuron : layer) {
                float[] weights = neuron.getWeights();
                for (int i = 0; i < weights.length; i++) {
                    Neuron input = neuron.getInputs().get(i);
                    connections.computeIfAbsent(input, k -> new ArrayList<>()).add(Pair.of(neuron, weights[i]));
                }
            }
        }
        return connections;
    }
}
