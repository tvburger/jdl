package net.tvburger.jdl.learning.loss;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.learning.LossFunction;
import net.tvburger.jdl.nn.NeuralNetwork;
import net.tvburger.jdl.nn.activations.Linear;
import net.tvburger.jdl.nn.activations.ReLU;
import net.tvburger.jdl.utils.Floats;

public class MSE implements LossFunction {

    private final float scale;

    public MSE(float scale) {
        this.scale = scale;
    }

    public float getScale() {
        return scale;
    }

    @Override
    public float calculateLoss(float error) {
        return scale * error * error;
    }

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, NeuralNetwork neuralNetwork) {
        float[] outputErrors = new float[neuralNetwork.getWidth(neuralNetwork.getDepth())];
        int totalSamples = dataSet.samples().size();
        for (DataSet.Sample sample : dataSet.samples()) {
            float[] predictedOutputs = neuralNetwork.estimate(sample.features());
            for (int i = 0; i < predictedOutputs.length; i++) {
                outputErrors[i] += (predictedOutputs[i] - sample.targetOutputs()[i]) / totalSamples;
            }
        }
        return outputErrors;
    }

    @Override
    public float[] determineGradients(DataSet dataSet, NeuralNetwork neuralNetwork) {
        if (!(neuralNetwork.getOutputActivationFunction() instanceof ReLU)
                && !(neuralNetwork.getOutputActivationFunction() instanceof Linear)) {
            throw new IllegalArgumentException("Only supported with ReLU and Linear");
        }
        float[] derivativesHalf = calculateOutputErrors(dataSet, neuralNetwork);
        if (!Floats.equals(scale, 0.5f)) {
            for (int i = 0; i < derivativesHalf.length; i++) {
                derivativesHalf[i] = derivativesHalf[i] * 2.0f * scale;
            }
        }
        return derivativesHalf;
    }

}
