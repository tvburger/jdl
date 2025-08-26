package net.tvburger.jdl.learning;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.nn.NeuralNetwork;

public interface Trainer {

    void train(NeuralNetwork neuralNetwork, DataSet trainingSet);

}
