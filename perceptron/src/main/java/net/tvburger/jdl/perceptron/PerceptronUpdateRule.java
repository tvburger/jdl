package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.LastInputStoredNeuron;
import net.tvburger.jdl.model.scalars.NeuronFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

public class PerceptronUpdateRule implements Optimizer.Stochastic<Perceptron, Float> {

    @Override
    public void optimize(Perceptron perceptron, DataSet.Sample<Float> sample, ObjectiveFunction<Float> objective, int step) {
        Array<Float> estimate = perceptron.estimate(sample.features());
        for (int i = 0; i < estimate.length(); i++) {
            int sign = Floats.greaterThan(sample.targetOutputs().get(i), 0.0f) ? +1 : -1;
            LastInputStoredNeuron neuron = perceptron.getNeuron(2, i, LastInputStoredNeuron.class);
            if (!Floats.equals(sample.targetOutputs().get(i), estimate.get(i))) {
                updateParameters(neuron, sign);
            }
        }
    }

    private void updateParameters(LastInputStoredNeuron neuron, float y) {
        NeuronFunction neuronFunction = neuron.getNeuronFunction();
        neuronFunction.adjustParameter(0, y);
        for (int d = 1; d <= neuron.arity(); d++) {
            neuronFunction.adjustParameter(d, y * neuron.getStoredInput(d));
        }
    }
}
