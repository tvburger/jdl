package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.learning.Trainer;
import net.tvburger.jdl.nn.NeuralNetwork;
import net.tvburger.jdl.nn.Neuron;
import net.tvburger.jdl.utils.Floats;

public class PerceptronTrainer implements Trainer {

    @Override
    public void train(NeuralNetwork perceptron, DataSet trainingSet) {
        if (trainingSet == null || trainingSet.samples() == null || trainingSet.samples().isEmpty()) {
            throw new IllegalArgumentException("No samples or training set specified!");
        }
        if (perceptron.getWidth(0) != trainingSet.samples().getFirst().features().length) {
            throw new IllegalArgumentException("Features are not matching input layer size!");
        }
        if (perceptron.getWidth(perceptron.getDepth()) != trainingSet.samples().getFirst().targetOutputs().length) {
            throw new IllegalArgumentException("Target outputs are not matching output layer size!");
        }
        for (DataSet.Sample sample : trainingSet.samples()) {
            float[] estimate = perceptron.estimate(sample.features());
            for (int i = 0; i < estimate.length; i++) {
                int sign = estimate[i] >= 0.0f ? +1 : -1;
                if (Floats.equals(sample.targetOutputs()[i], estimate[i])) {
                    continue;
                }
                updateParameters(perceptron.getNeuron(2, i), sign);
            }
        }
    }

    private void updateParameters(Neuron neuron, float y) {
        neuron.setBias(neuron.getBias() + y);
        float[] weights = neuron.getWeights();
        float[] storedInputs = neuron.getStoredInputs();
        for (int i = 0; i < weights.length; i++) {
            weights[i] += y * storedInputs[i] / neuron.getTotalActivations();
        }
        neuron.reset();
    }

}
