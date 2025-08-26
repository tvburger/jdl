package net.tvburger.jdl.nn.initializers;

import net.tvburger.jdl.nn.Initializer;
import net.tvburger.jdl.nn.Neuron;

import java.util.Random;

public class RandomInitializer implements Initializer {

    private final Random random = new Random();

    @Override
    public void initialize(Neuron neuron) {
        float[] weights = neuron.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random() / weights.length - 0.1f;
        }
    }

    private float random() {
        return -1.0f + 2.0f * random.nextFloat();
    }

}
