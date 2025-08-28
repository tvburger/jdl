package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.patterns.StaticUtility;

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

}
