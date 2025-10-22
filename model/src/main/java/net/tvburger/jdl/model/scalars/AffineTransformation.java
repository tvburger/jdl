package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * A simple linear scalar model of the form:
 * <pre>
 *   f(x) = bias + Σ w_i * x_i
 * </pre>
 * where the bias and weights are trainable parameters.
 * <p>
 * This class supports forward estimation of the scalar output as well as
 * calculation of parameter parameterGradients required for gradient-based training.
 */
@Strategy(Strategy.Role.CONCRETE)
public class AffineTransformation<N extends Number> extends LinearCombination<N> {

    private final JavaNumberTypeSupport<N> typeSupport;
    private Array<N> parameters;

    /**
     * Creates a new {@code AffineTransformation} with the specified number of input dimensions.
     * <p>
     * The underlying parameter vector will have length {@code dimensions + 1},
     * with the first element reserved for the bias and the rest for the weights.
     *
     * @param dimensions number of input features
     * @return a new {@code AffineTransformation} instance with zero-initialized parameters
     */
    public static <N extends Number> AffineTransformation<N> create(int dimensions, JavaNumberTypeSupport<N> typeSupport) {
        return new AffineTransformation<>(typeSupport.createArray(dimensions + 1), typeSupport);
    }


    /**
     * Constructs a {@code AffineTransformation} with the given parameter vector.
     * <p>
     * The first element of the array is treated as the bias, while the subsequent
     * elements are the weights.
     *
     * @param parameters the parameters
     */
    public AffineTransformation(Array<N> parameters, JavaNumberTypeSupport<N> typeSupport) {
        super(parameters.slice(1), typeSupport);
        this.typeSupport = typeSupport;
        this.parameters = parameters;
    }

    /**
     * Estimates the scalar output of this affine transformation for the given input vector.
     * <p>
     * The computation is:
     * <pre>
     *   output = bias + Σ (inputs[i] * weights[i])
     * </pre>
     *
     * @param inputs the input feature vector
     * @return the scalar output value
     * @throws IllegalArgumentException if {@code inputs.length != arity()}
     */
    @Override
    public N estimateScalar(Array<N> inputs) {
        return typeSupport.add(getBias(), super.estimateScalar(inputs));
    }

    /**
     * Calculates the parameterGradients of the output with respect to all parameters (bias and weights)
     * for the given input vector.
     * <p>
     * The gradient vector has length {@code 1 + arity()}, where:
     * <ul>
     *   <li>Index 0 corresponds to the derivative w.r.t. the bias (always {@code 1}).</li>
     *   <li>Indices 1..arity correspond to the input values, i.e. ∂f/∂w_i = input[i-1].</li>
     * </ul>
     *
     * @param inputs the input feature vector
     * @return the gradient vector (bias gradient followed by weight parameterGradients)
     */
    @Override
    public Array<N> calculateParameterGradients(Array<N> inputs) {
        Array<N> gradients = typeSupport.createArray(1 + arity());
        gradients.set(0, typeSupport.one());
        if (arity() > 0) {
            gradients.set(super.calculateParameterGradients(inputs), 1);
        }
        return gradients;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getParameterCount() {
        return super.getParameterCount() + 1;
    }

    /**
     * Returns the full parameter vector of this model.
     * The first element is the bias, the rest are the weights.
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
        parameters = values;
        super.setParameters(values.slice(1));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setParameter(int p, N value) {
        parameters.set(p, value);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void adjustParameters(Array<N> deltas) {
        for (int p = 0; p < parameters.length(); p++) {
            adjustParameter(p, deltas.get(p));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void adjustParameter(int p, N delta) {
        parameters.set(p, typeSupport.add(parameters.get(p), delta));
    }

    /**
     * Returns the bias parameter.
     *
     * @return the bias value (0.0f if there are no parameters)
     */
    public N getBias() {
        return parameters.get(0);
    }

    /**
     * Sets the bias parameter.
     *
     * @param bias the new bias value
     */
    public void setBias(N bias) {
        parameters.set(0, bias);
    }

    /**
     * Adjusts the bias parameter by adding {@code delta}.
     *
     * @param delta the adjustment value
     */
    public void adjustBias(N delta) {
        adjustParameter(0, delta);
    }

}
