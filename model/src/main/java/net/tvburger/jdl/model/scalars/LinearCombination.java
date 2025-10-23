package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.model.ParameterizedTrainableScalarEstimationFunction;
import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.numbers.Tensor;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * A simple linear combination of the form:
 * <pre>
 *   f(x) = Σ w_i * x_i
 * </pre>
 * where the weights are trainable parameters.
 * <p>
 * This class supports forward estimation of the scalar output as well as
 * calculation of parameter parameterGradients required for gradient-based training.
 *
 * @see AffineTransformation
 */
@Strategy(Strategy.Role.CONCRETE)
public class LinearCombination<N extends Number> implements ParameterizedTrainableScalarEstimationFunction<N, N, N> {

    private final JavaNumberTypeSupport<N> typeSupport;
    private Array<N> parameters;

    /**
     * Creates a new {@code LinearCombination} with the specified number of input dimensions.
     * <p>
     * The underlying parameter vector will have length {@code dimensions} representing the weights.
     *
     * @param dimensions number of input features
     * @return a new {@code LinearCombination} instance with zero-initialized parameters
     */
    public static <N extends Number> LinearCombination<N> create(int dimensions, JavaNumberTypeSupport<N> typeSupport) {
        return new LinearCombination<>(typeSupport.createArray(dimensions), typeSupport);
    }

    /**
     * Constructs a {@code LinearCombination} with the given parameter vector.
     *
     * @param parameters parameter vector (weights)
     */
    public LinearCombination(Array<N> parameters, JavaNumberTypeSupport<N> typeSupport) {
        this.parameters = parameters;
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }

    /**
     * Estimates the scalar output of this linear model for the given input vector.
     * <p>
     * The computation is:
     * <pre>
     *   output = Σ (inputs[i] * weights[i])
     * </pre>
     *
     * @param inputs the input feature vector
     * @return the scalar output value
     * @throws IllegalArgumentException if {@code inputs.length != arity()}
     */
    @Override
    public N estimateScalar(Array<N> inputs) {
        if (inputs.length() != parameters.length()) {
            throw new IllegalArgumentException();
        }
        N sum = typeSupport.zero();
        for (int d = 1; d <= parameters.length(); d++) {
            sum = typeSupport.add(sum, typeSupport.multiply(inputs.get(d - 1), getWeight(d)));
        }
        return sum;
    }

    /**
     * Calculates the parameterGradients of the output with respect to all parameters (weights)
     * for the given input vector.
     * <p>
     * The gradient vector has length {@code arity()}, where:
     * <ul>
     *   <li>Indices 0..arity correspond to the input values, i.e. ∂f/∂w_i = input[i-1].</li>
     * </ul>
     *
     * @param inputs the input feature vector
     * @return the gradient vector (weight parameterGradients)
     */
    @Override
    public Array<N> getParameterGradients(Array<N> inputs) {
        return inputs;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getParameterCount() {
        return parameters.length();
    }

    /**
     * Returns the full parameter vector of this model.
     *
     * @return the parameter array
     */
    @Override
    public Array<N> getParameters() {
        return parameters;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N getParameter(int p) {
        return parameters.get(p);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setParameters(Array<N> values) {
        this.parameters = values;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setParameter(int p, N value) {
        parameters.set(p, value);
    }

    /**
     * Returns the number of input features this model accepts,
     * equal to the number of weights (parameters.length).
     *
     * @return the arity (number of input dimensions)
     */
    @Override
    public int arity() {
        return parameters.length();
    }

    /**
     * Checks if a given dimension index is valid.
     *
     * @param d dimension index, starting from 1
     * @return the same index if valid
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    protected final int validDimension(int d) {
        if (d < 1 || d > parameters.length()) {
            throw new IllegalArgumentException("Invalid dimension!");
        }
        return d;
    }

    /**
     * Returns a copy of the weights (all parameters except the bias).
     *
     * @return the weight vector
     */
    public Array<N> getWeights() {
        return parameters;
    }

    /**
     * Replaces all weights with the given vector.
     *
     * @param weights the new weight vector
     * @throws IllegalArgumentException if the number of weights does not match {@link #arity()}
     */
    public void setWeights(Array<N> weights) {
        if (weights.length() != parameters.length()) {
            throw new IllegalArgumentException("Invalid number of weights!");
        }
        this.parameters = weights;
    }

    /**
     * Returns the weight for a given dimension (1-based).
     *
     * @param d dimension index, starting from 1
     * @return the weight value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public N getWeight(int d) {
        return parameters.get(validDimension(d) - 1);
    }

    /**
     * Sets the weight for a given dimension (1-based).
     *
     * @param d      dimension index, starting from 1
     * @param weight the new weight value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public void setWeight(int d, N weight) {
        parameters.set(validDimension(d) - 1, weight);
    }

    /**
     * Adjusts the weight for a given dimension (1-based) by adding {@code delta}.
     *
     * @param d     dimension index, starting from 1
     * @param delta the adjustment value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public void adjustWeight(int d, N delta) {
        parameters.set(d - 1, typeSupport.add(parameters.get(validDimension(d) - 1), delta));
    }

    @Override
    public Tensor<N> mapToTensor(Tensor<N> tensor) {
        return null;
    }
}
