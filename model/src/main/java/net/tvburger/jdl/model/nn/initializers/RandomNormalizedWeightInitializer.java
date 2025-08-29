package net.tvburger.jdl.model.nn.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.Random;

/**
 * Implements a random initializer. It sets every weight parameter of the model to a random value between [-1/#w, 1/#w]
 * where #w is the number of weights.
 * <p>
 * The bias is left unchanged.
 */
@Strategy(role = Strategy.Role.CONCRETE)
public class RandomNormalizedWeightInitializer implements Initializer {

    private final Random random = new Random();

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(Neuron neuron) {
        float[] weights = neuron.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random() / weights.length;
        }
    }

    private float random() {
        return -1.0f + 2.0f * random.nextFloat();
    }

}
