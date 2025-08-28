package net.tvburger.jdl.adaline;

import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.initializers.Initializer;

import java.util.Random;

public class AdalineInitializer implements Initializer {

    @Override
    public void initialize(Neuron neuron) {
        Random random = new Random();
        float[] weights = neuron.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (random.nextFloat() - 0.5f) * 0.01f;
        }
        neuron.setBias(0.0f);
    }
}
