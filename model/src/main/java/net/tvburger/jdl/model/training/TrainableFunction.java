package net.tvburger.jdl.model.training;

import net.tvburger.jdl.model.EstimationFunction;

/**
 * A function with trainable parameters that can be estimated and updated.
 * <p>
 * Extends {@link EstimationFunction} by adding accessors and mutators for
 * parameters (biases, weights, coefficients, etc.) used during training.
 * Provides default convenience methods for reading, writing, and updating
 * parameters.
 */
public interface TrainableFunction<N extends Number> extends EstimationFunction<N> {

    /**
     * Returns the number of trainable parameters of this function.
     *
     * @return the number of parameters
     */
    int getParameterCount();

    /**
     * Returns the underlying parameter vector for this function.
     *
     * @return the array of parameters
     */
    N[] getParameters();

    /**
     * Returns the parameter at the given index.
     *
     * @param p index of the parameter
     * @return the parameter value
     * @throws ArrayIndexOutOfBoundsException if the index is out of range
     */
    N getParameter(int p);

    /**
     * Replaces the parameter vector with the given values.
     *
     * @param values the new parameter values
     * @throws IllegalArgumentException if the array length does not match
     *                                  {@link #getParameterCount()}
     */
    void setParameters(N[] values);

    /**
     * Sets the parameter at the given index to the specified value.
     *
     * @param p     index of the parameter
     * @param value new value to set
     * @throws ArrayIndexOutOfBoundsException if the index is out of range
     */
    void setParameter(int p, N value);

    /**
     * Adjusts all parameters (except the first one, typically the bias)
     * by adding the corresponding deltas.
     *
     * @param deltas an array of additive updates, aligned with the parameter vector
     */
    default void adjustParameters(N[] deltas) {
        for (int i = 0; i < deltas.length; i++) {
            adjustParameter(i, deltas[i]);
        }
    }

    /**
     * Adjusts a single parameter by adding the given delta to its current value.
     *
     * @param p     index of the parameter
     * @param delta the increment to add to the parameter value
     * @throws ArrayIndexOutOfBoundsException if the index is out of range
     */
    default void adjustParameter(int p, N delta) {
        setParameter(p, getCurrentNumberType().add(getParameter(p), delta));
    }

}
