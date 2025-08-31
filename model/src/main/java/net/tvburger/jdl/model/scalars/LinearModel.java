package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.patterns.TemplateMethod;
import net.tvburger.jdl.common.utils.Floats;

import java.util.Arrays;

/**
 * Abstract base class for linear scalar models of the form:
 * <pre>
 *   f(x) = bias + Σ w_i * x_i
 * </pre>
 * where parameters[0] is the bias and parameters[1..] are the weights.
 * <p>
 * Implements {@link TrainableScalarFunction}, exposing direct accessors and
 * mutators for weights and bias, as well as parameter adjustments used during training.
 */
@TemplateMethod
public abstract class LinearModel implements TrainableScalarFunction {

    private final float[] parameters;

    /**
     * Creates a new linear model with the given parameter vector.
     * <p>
     * The first element is the bias, subsequent elements are the weights.
     *
     * @param parameters bias and weight parameters of the model
     */
    public LinearModel(float[] parameters) {
        this.parameters = parameters;
    }

    /**
     * Returns the full parameter vector of this model.
     * The first element is the bias, the rest are the weights.
     *
     * @return the parameter array
     */
    @Override
    public float[] getParameters() {
        return parameters;
    }

    /**
     * Returns the number of input features this model accepts,
     * equal to the number of weights (parameters.length - 1).
     *
     * @return the arity (number of input dimensions)
     */
    @Override
    public int arity() {
        return Math.max(0, parameters.length - 1);
    }

    /**
     * Checks if a given dimension index is valid.
     *
     * @param d dimension index, starting from 1
     * @return the same index if valid
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    protected final int validDimension(int d) {
        if (d < 1 || d > arity()) {
            throw new IllegalArgumentException("Invalid dimension!");
        }
        return d;
    }

    /**
     * Returns a copy of the weights (all parameters except the bias).
     *
     * @return the weight vector
     */
    public float[] getWeights() {
        return arity() > 0 ? Arrays.copyOfRange(parameters, 1, arity() + 1) : Floats.EMPTY;
    }

    /**
     * Replaces all weights with the given vector.
     *
     * @param weights the new weight vector
     * @throws IllegalArgumentException if the number of weights does not match {@link #arity()}
     */
    public void setWeights(float[] weights) {
        if (weights.length != coArity()) {
            throw new IllegalArgumentException("Invalid number of weights!");
        }
        System.arraycopy(weights, 0, parameters, 1, arity());
    }

    /**
     * Returns the weight for a given dimension (1-based).
     *
     * @param d dimension index, starting from 1
     * @return the weight value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public float getWeight(int d) {
        return parameters[validDimension(d)];
    }

    /**
     * Sets the weight for a given dimension (1-based).
     *
     * @param d      dimension index, starting from 1
     * @param weight the new weight value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public void setWeight(int d, float weight) {
        parameters[validDimension(d)] = weight;
    }

    /**
     * Adjusts the weight for a given dimension (1-based) by adding {@code delta}.
     *
     * @param d     dimension index, starting from 1
     * @param delta the adjustment value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public void adjustWeight(int d, float delta) {
        parameters[validDimension(d)] += delta;
    }

    /**
     * Returns the bias parameter.
     *
     * @return the bias value (0.0f if there are no parameters)
     */
    public float getBias() {
        return parameters.length == 0 ? 0.0f : parameters[0];
    }

    /**
     * Sets the bias parameter.
     *
     * @param bias the new bias value
     */
    public void setBias(float bias) {
        parameters[0] = bias;
    }

    /**
     * Adjusts the bias parameter by adding {@code delta}.
     *
     * @param delta the adjustment value
     */
    public void adjustBias(float delta) {
        parameters[0] += delta;
    }
}
