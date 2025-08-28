package net.tvburger.jdl.model.nn.initializers;

import net.tvburger.jdl.common.patterns.StaticUtility;

/**
 * Utility for obtaining (singleton) instances of initializers.
 */
@StaticUtility
public final class Initializers {

    private static final RandomWeightInitializer random = new RandomWeightInitializer();

    private Initializers() {
    }

    /**
     * Returns the random weight initializer
     *
     * @return the random weight initializer
     */
    public static RandomWeightInitializer randomWeight() {
        return random;
    }

}
