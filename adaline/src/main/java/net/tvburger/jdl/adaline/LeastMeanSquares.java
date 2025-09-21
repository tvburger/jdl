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
        float[] gradients = objective.calculateGradient_dJ_da(1, estimated, sample.targetOutputs());
        for (int j = 0; j < estimated.length; j++) {
            Neuron node = adaline.getNeuron(adaline.getDepth(), j, Neuron.class);
            float errorSignal = gradients[j];
            for (int d = 1; d <= sample.featureCount(); d++) {
                node.getNeuronFunction().adjustParameter(d, learningRate * errorSignal * sample.features()[d - 1]);
            }
            node.adjustParameter(0, learningRate * errorSignal);
        }
    }
}
