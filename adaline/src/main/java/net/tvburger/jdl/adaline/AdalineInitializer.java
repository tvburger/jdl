package net.tvburger.jdl.adaline;

import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.training.initializers.NeuralNetworkInitializer;

import java.util.Random;

public class AdalineInitializer implements NeuralNetworkInitializer {

    private final Random random = new Random();

    @Override
    public void initialize(Neuron neuron) {
        for (int d = 1; d <= neuron.arity(); d++) {
            neuron.setWeight(1, (random.nextFloat() - 0.5f) * 0.01f);
        }
        neuron.setBias(0.0f);
    }
}
