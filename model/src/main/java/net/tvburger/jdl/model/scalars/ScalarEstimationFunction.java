package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * A specialization of {@link EstimationFunction} for functions that produce
 * a single scalar output.
 * <p>
 * Provides a scalar-valued estimation method, while also offering a default
 * vector-based interface compatible with {@link EstimationFunction}.
 */
@Strategy(Strategy.Role.INTERFACE)
public interface ScalarEstimationFunction<N extends Number> extends EstimationFunction<N> {

    /**
     * Estimates the output of this function for the given input vector,
     * returning the result as a single-element array.
     *
     * @param inputs the input feature vector
     * @return a single-element array containing {@link #estimateScalar(N[])}
     */
    default N[] estimate(N[] inputs) {
        N[] array = getCurrentNumberType().createArray(1);
        array[0] = estimateScalar(inputs);
        return array;
    }

    /**
     * Estimates the scalar output of this function for the given input vector.
     *
     * @param inputs the input feature vector
     * @return the scalar output value
     */
    N estimateScalar(N[] inputs);

    /**
     * Returns the number of output values (co-arity) of this function.
     * For a scalar estimation function, this is always {@code 1}.
     *
     * @return the co-arity, always {@code 1}
     */
    default int coArity() {
        return 1;
    }
}
