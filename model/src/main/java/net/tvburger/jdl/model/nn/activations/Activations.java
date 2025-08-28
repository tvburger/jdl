package net.tvburger.jdl.model.nn.activations;

import net.tvburger.jdl.common.patterns.StaticUtility;

/**
 * Utility for obtaining (singleton) instances of activation functions.
 */
@StaticUtility
public final class Activations {

    private static final ReLU relu = new ReLU();
    private static final Linear linear = new Linear();
    private static final Sigmoid sigmoid = new Sigmoid();
    private static final Step step = new Step();

    private Activations() {
    }

    /**
     * Returns the ReLU.
     *
     * @return the ReLU
     */
    public static ReLU reLU() {
        return relu;
    }

    /**
     * Returns the linear activation (logit = output)
     *
     * @return the linear activation
     */
    public static Linear linear() {
        return linear;
    }

    /**
     * Returns the linear activation as it is the identity activation function
     *
     * @return the linear activation
     */
    public static Linear identity() {
        return linear();
    }

    /**
     * Returns the linear activation as it is the same having no activation function
     *
     * @return the linear activation
     */
    public static Linear none() {
        return linear();
    }

    /**
     * Returns the sigmoid activation
     *
     * @return sigmoid activation
     */
    public static Sigmoid sigmoid() {
        return sigmoid;
    }

    /**
     * Returns the step activation. Note that if you configure this step function, you configure a global one, applied
     * to all the steps returned by this function.
     *
     * @return the step activation
     */
    public static Step step() {
        return step;
    }
}
