package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.nn.Initializer;
import net.tvburger.jdl.nn.InputNeuron;
import net.tvburger.jdl.nn.Neuron;

import java.util.Random;

public class PerceptronInitializer implements Initializer {

    private final Random random = new Random();

    @Override
    public void initialize(Neuron neuron) {
        if (neuron instanceof InputNeuron) {
            return;
        }
        float[] weights = neuron.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random();
        }
        if (neuron instanceof AssociationUnit) {
            neuron.setBias(random());
        }
    }

    private float random() {
        return -1.0f + 2.0f * random.nextFloat();
    }
}
