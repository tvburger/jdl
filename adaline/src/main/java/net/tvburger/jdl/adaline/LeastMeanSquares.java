package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

public class LeastMeanSquares implements Optimizer.Stochastic<Adaline, Float> {

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

    @Override
    public void optimize(Adaline adaline, DataSet.Sample<Float> sample, ObjectiveFunction<Float> objective, int step) {
        Array<Float> estimated = adaline.estimate(sample.features());
        Array<Float> gradients = objective.calculateGradient_dJ_da(1, estimated, sample.targetOutputs());
        for (int j = 0; j < estimated.length(); j++) {
            Neuron node = adaline.getNeuron(adaline.getDepth(), j, Neuron.class);
            float errorSignal = -1 * gradients.get(j);
            for (int d = 1; d <= sample.featureCount(); d++) {
                node.getNeuronFunction().adjustParameter(d, learningRate * errorSignal * sample.features().get(d - 1));
            }
            node.adjustParameter(0, learningRate * errorSignal);
        }
    }
}
