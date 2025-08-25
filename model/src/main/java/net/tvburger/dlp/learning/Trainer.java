package net.tvburger.dlp.learning;

import net.tvburger.dlp.DataSet;
import net.tvburger.dlp.nn.NeuralNetwork;

public interface Trainer {

    void train(NeuralNetwork neuralNetwork, DataSet trainingSet);

}
