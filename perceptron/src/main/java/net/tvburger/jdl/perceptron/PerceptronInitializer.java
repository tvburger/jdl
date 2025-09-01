package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.model.nn.InputNeuron;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.training.initializers.NeuralNetworkInitializer;

import java.util.Random;

public class PerceptronInitializer implements NeuralNetworkInitializer {

    private final Random random = new Random();

    @Override
    public void initialize(Neuron neuron) {
        if (neuron instanceof InputNeuron) {
            return;
        }
        for (int d = 1; d <= neuron.arity(); d++) {
            neuron.setWeight(d, random());
        }
        if (neuron instanceof AssociationUnit) {
            neuron.setBias(random());
        }
    }

    private float random() {
        return -1.0f + 2.0f * random.nextFloat();
    }
}
