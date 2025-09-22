package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.LastInputStoredNeuron;
import net.tvburger.jdl.model.scalars.NeuronFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

import java.util.Arrays;

public class PerceptronUpdateRule implements Optimizer.OnlineOnly<Perceptron, Float> {

    @Override
    public void optimize(Perceptron perceptron, DataSet.Sample<Float> sample, ObjectiveFunction objective) {
        Float[] estimate = perceptron.estimate(sample.features());
        for (int i = 0; i < estimate.length; i++) {
            int sign = Floats.greaterThan(sample.targetOutputs()[i], 0.0f) ? +1 : -1;
            LastInputStoredNeuron neuron = perceptron.getNeuron(2, i, LastInputStoredNeuron.class);
            if (!Floats.equals(sample.targetOutputs()[i], estimate[i])) {
                System.out.println("Training: real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate) + ": features = " + Arrays.toString(sample.features()));
                System.out.println("Sample target output = " + sample.targetOutputs()[i] + ", sign = " + sign);
                updateParameters(neuron, sign);
            }
        }
    }

    private void updateParameters(LastInputStoredNeuron neuron, float y) {
        NeuronFunction neuronFunction = neuron.getNeuronFunction();
        neuronFunction.adjustParameter(0, y);
        for (int d = 1; d < neuron.arity(); d++) {
            neuronFunction.adjustParameter(d, y * neuron.getStoredInput(d));
        }
        System.out.println(neuron);
    }
}
