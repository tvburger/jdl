package net.tvburger.dlp.initializers;

import net.tvburger.dlp.nn.Initializer;
import net.tvburger.dlp.nn.Neuron;

import java.util.Random;

public class RandomInitializer implements Initializer {

    private final Random random = new Random();

    @Override
    public void initialize(Neuron neuron) {
        float[] weights = neuron.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 1.0f * (random.nextInt(100)) / 100f;
        }
    }
}
