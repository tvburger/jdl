package net.tvburger.jdl.model.nn.training.initializers;

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

    private final Random random;

    /**
     * Creates a new Xavier initializer with a default, non-deterministic
     * random number generator.
     * <p>
     * This constructor is suitable when reproducibility is not required.
     * Each run will produce different initial weights.
     * </p>
     */
    public XavierInitializer() {
        this(new Random());
    }

    /**
     * Creates a new Xavier initializer with a seeded random number generator.
     * <p>
     * Using a fixed seed makes the initializer deterministic, ensuring that
     * the same sequence of random numbers (and thus weight initializations)
     * is produced across runs.
     * </p>
     *
     * @param seed the seed for the random number generator
     */
    public XavierInitializer(int seed) {
        this(new Random(seed));
    }

    /**
     * Creates a new Xavier initializer with the given {@link Random} instance.
     * <p>
     * This constructor provides maximum flexibility by allowing the caller to
     * supply a custom random number generator implementation or one shared with
     * other components.
     * </p>
     *
     * @param random the random number generator to use
     */
    public XavierInitializer(Random random) {
        this.random = random;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(NeuralNetwork neuralNetwork, Neuron neuron) {
        int fanIn = neuron.arity();
        int fanOut = neuralNetwork.getTargetNeurons(neuron).size();

        float limit = (float) Math.sqrt(6.0f / (fanIn + fanOut));

        for (int d = 1; d <= neuron.arity(); d++) {
            neuron.setWeight(d, uniform(-limit, limit));
        }
        // Bias left unchanged
    }

    private float uniform(float min, float max) {
        return min + (max - min) * random.nextFloat();
    }
}