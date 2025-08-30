package net.tvburger.jdl.model.nn.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.Random;

/**
 * Xavier (Glorot) weight initializer.
 * <p>
 * For sigmoid/tanh activations, initializes weights from a uniform
 * distribution in the range [-limit, limit], where:
 * <p>
 * limit = sqrt(6 / (fanIn + fanOut))
 * <p>
 * fanIn  = number of input connections (weights)
 * fanOut = number of output connections (neurons this neuron connects to)
 * <p>
 * Bias is left unchanged (usually set separately).
 */
@Strategy(Strategy.Role.CONCRETE)
public class XavierInitializer implements NeuralNetworkInitializer {

    private final Random random = new Random();

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(NeuralNetwork neuralNetwork, Neuron neuron) {
        float[] weights = neuron.getWeights();
        int fanIn = weights.length;
        int fanOut = neuralNetwork.getTargetNeurons(neuron).size();

        float limit = (float) Math.sqrt(6.0f / (fanIn + fanOut));

        for (int i = 0; i < weights.length; i++) {
            weights[i] = uniform(-limit, limit);
        }
        // Bias left unchanged
    }

    private float uniform(float min, float max) {
        return min + (max - min) * random.nextFloat();
    }
}