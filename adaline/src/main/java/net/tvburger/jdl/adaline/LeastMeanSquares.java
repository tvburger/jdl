package net.tvburger.jdl.adaline;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

public class LeastMeanSquares implements Optimizer.OnlineOnly<Adaline> {

    private float learningRate;

    public LeastMeanSquares(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public void optimize(Adaline adaline, DataSet.Sample sample, ObjectiveFunction objective) {
        float[] estimated = adaline.estimate(sample.features());
        float[] gradients = objective.determineGradients(estimated, sample.targetOutputs());
        for (int i = 0; i < estimated.length; i++) {
            Neuron node = adaline.getNeuron(adaline.getDepth(), i, Neuron.class);
            float errorSignal = gradients[i];
            for (int w = 0; w < sample.featureCount(); w++) {
                node.getWeights()[w] += learningRate * errorSignal * sample.features()[w];
            }
            node.setBias(node.getBias() + learningRate * errorSignal);
        }
    }
}
