package net.tvburger.jdl.model.scalars.activations;

import net.tvburger.jdl.common.function.UnaryFunction;
import net.tvburger.jdl.common.numbers.Scalar;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * An ActivationFunction maps a logit to an output.
 */
@Strategy(Strategy.Role.INTERFACE)
public interface ActivationFunction extends UnaryFunction<Float, Float> {

    /**
     * Name of the activation function.
     *
     * @return the name
     */
    default String name() {
        return getClass().getSimpleName();
    }

    /**
     * Maps the logit to the corresponding output.
     *
     * @param logit the logit to map
     * @return the corresponding output value
     */
    float activate(float logit);

    default Scalar<Float> apply(Scalar<Float> input) {
        return Scalar.of(activate(input.get()));
    }

    /**
     * Determine the slope (gradient) at the given output value
     *
     * @param output the output at which the gradient is determined
     * @return the gradient
     * @throws UnsupportedOperationException when the gradient is not supported by this function
     */
    default float determineGradientForOutput(float output) {
        throw new UnsupportedOperationException();
    }

}
