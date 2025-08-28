package net.tvburger.jdl.wip;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.learning.Trainer;
import net.tvburger.jdl.wip.loss.LossFunction;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.HashMap;
import java.util.Map;

public class GradientDescent<N extends NeuralNetwork> implements Trainer<N> {

    public static final float DEFAULT_LEARNING_RATE = 0.1f;

    private final LossFunction<N> lossFunction;
    private float learningRate = DEFAULT_LEARNING_RATE;

    public GradientDescent(LossFunction<N> lossFunction) {
        this.lossFunction = lossFunction;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public LossFunction<N> getLossFunction() {
        return lossFunction;
    }

    @Override
    public void train(N neuralNetwork, DataSet trainingSet) {
        float[] gradients = lossFunction.determineGradients(trainingSet, neuralNetwork);
        Map<Neuron, Float> errorSignals = new HashMap<>();
        for (int i = neuralNetwork.getDepth(); i > 0; i--) {
            int width = neuralNetwork.getWidth(i);
            for (int j = 0; j < width; j++) {
                Neuron neuron = neuralNetwork.getNeuron(i, j);
                float errorSignal;
                if (i == neuralNetwork.getDepth()) {
                    errorSignal = gradients[j];
                } else {
                    errorSignal = 0.0f;
                    Map<Neuron, Float> outputConnections = neuralNetwork.getOutputConnections(i, j);
                    for (Map.Entry<Neuron, Float> outputConnection : outputConnections.entrySet()) {
                        errorSignal += errorSignals.get(outputConnection.getKey()) * outputConnection.getValue();
                    }
                }
                errorSignals.put(neuron, errorSignal);
                updateParameters(neuron, errorSignal);
            }
        }
    }

    private void updateParameters(Neuron neuron, float errorSignal) {
        neuron.setBias(neuron.getBias() - errorSignal * learningRate);
//        float[] weights = neuron.getWeights();
//        float[] storedInputs = neuron.getStoredInputs();
//        for (int i = 0; i < weights.length; i++) {
//            weights[i] -= errorSignal * learningRate * storedInputs[i] / neuron.getTotalActivations();
//        }
//        neuron.reset();
    }

}
