package net.tvburger.jdl.model.nn.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.Random;

/**
 * He (Kaiming) weight initializer.
 * <p>
 * Initializes weights with values drawn from a Normal distribution
 * with mean 0 and variance 2/fanIn, where fanIn is the number of
 * input connections to the neuron. This works well with ReLU activations.
 * <p>
 * The bias is left unchanged (as is common practice).
 */
@Strategy(role = Strategy.Role.CONCRETE)
public class HeInitializer implements Initializer {

    private final Random random = new Random();

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(Neuron neuron) {
        float[] weights = neuron.getWeights();
        int fanIn = weights.length;
        double std = Math.sqrt(2.0 / fanIn);

        for (int i = 0; i < weights.length; i++) {
            // Gaussian(0, std^2)
            weights[i] = (float) (random.nextGaussian() * std);
        }
        // Bias left unchanged
    }
}