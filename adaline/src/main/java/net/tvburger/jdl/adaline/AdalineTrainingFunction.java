package net.tvburger.jdl.adaline;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.learning.OnlineTrainer;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.Arrays;

public class AdalineTrainingFunction implements OnlineTrainer.Function<Adaline> {

    private float learningRate;

    public AdalineTrainingFunction(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void train(Adaline adaline, DataSet.Sample sample) {
        float[] estimated = adaline.estimate(sample.features());
        for (int i = 0; i < estimated.length; i++) {
            Neuron node = adaline.getNeuron(adaline.getDepth(), i, Neuron.class);
            float error = sample.targetOutputs()[i] - estimated[i];
            for (int w = 0; w < sample.featureCount(); w++) {
                node.getWeights()[w] += learningRate * error * sample.features()[w];
            }
            node.setBias(node.getBias() + learningRate * error);
        }
    }

}
