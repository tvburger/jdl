package net.tvburger.jdl.model.nn.training.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.Random;

/**
 * Implements a random initializer. It sets every weight parameter of the model to a random value between [-1/#w, 1/#w]
 * where #w is the number of weights.
 * <p>
 * The bias is left unchanged.
 */
@Strategy(Strategy.Role.CONCRETE)
public class RandomNormalizedWeightInitializer implements NeuralNetworkInitializer {

    private final Random random = new Random();

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(Neuron neuron) {
        for (int d = 1; d <= neuron.arity(); d++) {
            neuron.setWeight(d, random() / neuron.arity());
        }
    }

    private float random() {
        return -1.0f + 2.0f * random.nextFloat();
    }

}
