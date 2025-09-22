package net.tvburger.jdl.model;

import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * The {@link #estimate(N[])} method defines the contract for mapping input values
 * to output values according to a specific estimation algorithm.
 * Different implementations may use linear, non-linear, or domain-specific (symbolic AI) logic.
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface EstimationFunction<N extends Number> extends NumberTypeAgnostic<N> {

    /**
     * Applies the estimation algorithm to the given input vector.
     *
     * @param inputs an array of input values (e.g., features, signals)
     * @return an array of output values computed by the estimation function
     */
    N[] estimate(N[] inputs);

    /**
     * Returns the <b>arity</b> of this function, i.e. the number of inputs
     * (independent variables or features) it expects.
     * <p>
     * For example:
     * <ul>
     *   <li>A univariate function {@code f(x)} has arity 1.</li>
     *   <li>A multivariate function {@code f(x, y, z)} has arity 3.</li>
     *   <li>In machine learning, this corresponds to the input feature dimension.</li>
     * </ul>
     *
     * @return the number of inputs (arity) this function accepts
     */
    int arity();


    /**
     * Returns the <b>co-arity</b> of this function, i.e. the number of outputs
     * (dependent variables or results) it produces.
     * <p>
     * For example:
     * <ul>
     *   <li>A scalar-valued function {@code f(x, y)} → ℝ has co-arity 1.</li>
     *   <li>A vector-valued function {@code f(x, y)} → ℝ² has co-arity 2.</li>
     *   <li>In machine learning, this corresponds to the output dimension
     *       (e.g., number of labels or regression targets).</li>
     * </ul>
     *
     * @return the number of outputs (co-arity) this function produces
     */
    int coArity();

}
