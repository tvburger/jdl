package net.tvburger.jdl.model.nn.training.initializers;

import net.tvburger.jdl.common.patterns.StaticUtility;

/**
 * Utility for obtaining (singleton) instances of initializers.
 */
@StaticUtility
public final class NeuralNetworkInitializers {

    private static final RandomNormalizedWeightInitializer random = new RandomNormalizedWeightInitializer();
    private static final HeInitializer he = new HeInitializer();
    private static final XavierInitializer xavier = new XavierInitializer();

    private NeuralNetworkInitializers() {
    }

    /**
     * Returns the random weight initializer
     *
     * @return the random weight initializer
     */
    public static RandomNormalizedWeightInitializer randomWeight() {
        return random;
    }

    /**
     * Returns the Xavier initializer
     *
     * @return the Xavier initializer
     */
    public static XavierInitializer xavier() {
        return xavier;
    }

    /**
     * Returns the He initializer
     *
     * @return the He initializer
     */
    public static HeInitializer he() {
        return he;
    }

    /**
     * Returns a constant initializer
     *
     * @param constant the constant used for initialization
     * @return a constant initializer
     */
    public static ConstantInitializer constant(float constant) {
        return new ConstantInitializer(constant);
    }
}
