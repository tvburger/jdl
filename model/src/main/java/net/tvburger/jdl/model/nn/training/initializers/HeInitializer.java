package net.tvburger.jdl.model.nn.training.initializers;

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
@Strategy(Strategy.Role.CONCRETE)
public class HeInitializer implements NeuralNetworkInitializer {

    private final Random random = new Random();

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(Neuron neuron) {
        int fanIn = neuron.arity();
        double std = Math.sqrt(2.0 / fanIn);

        for (int d = 1; d <= neuron.arity(); d++) {
            // Gaussian(0, std^2)
            neuron.setParameter(d, (float) (random.nextGaussian() * std));
        }
        // Bias left unchanged
    }
}