package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.Arrays;

/**
 * A simple linear scalar model of the form:
 * <pre>
 *   f(x) = bias + Σ w_i * x_i
 * </pre>
 * where the bias and weights are trainable parameters.
 * <p>
 * This class supports forward estimation of the scalar output as well as
 * calculation of parameter gradients required for gradient-based training.
 */
@Strategy(Strategy.Role.CONCRETE)
public class LinearCombination<N extends Number> implements TrainableScalarFunction<N> {

    private final N[] parameters;
    private final JavaNumberTypeSupport<N> typeSupport;

    /**
     * Creates a new {@code LinearCombination} with the specified number of input dimensions.
     * <p>
     * The underlying parameter vector will have length {@code dimensions + 1},
     * with the first element reserved for the bias and the rest for the weights.
     *
     * @param dimensions number of input features
     * @return a new {@code LinearCombination} instance with zero-initialized parameters
     */
    public static <N extends Number> LinearCombination<N> create(int dimensions, JavaNumberTypeSupport<N> typeSupport) {
        return new LinearCombination<>(typeSupport.createArray(dimensions + 1), typeSupport);
    }


    /**
     * Constructs a {@code LinearCombination} with the given parameter vector.
     * <p>
     * The first element of the array is treated as the bias, while the subsequent
     * elements are the weights.
     *
     * @param parameters parameter vector (bias + weights)
     */
    public LinearCombination(N[] parameters, JavaNumberTypeSupport<N> typeSupport) {
        this.parameters = parameters;
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return typeSupport;
    }

    /**
     * Estimates the scalar output of this linear model for the given input vector.
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
    public N estimateScalar(N[] inputs) {
        if (inputs.length != arity()) {
            throw new IllegalArgumentException();
        }
        N sum = getBias();
        for (int d = 1; d <= arity(); d++) {
            sum = typeSupport.add(sum, typeSupport.multiply(inputs[d - 1], getWeight(d)));
        }
        return sum;
    }

    /**
     * Calculates the gradients of the output with respect to all parameters (bias and weights)
     * for the given input vector.
     * <p>
     * The gradient vector has length {@code 1 + arity()}, where:
     * <ul>
     *   <li>Index 0 corresponds to the derivative w.r.t. the bias (always {@code 1}).</li>
     *   <li>Indices 1..arity correspond to the input values, i.e. ∂f/∂w_i = input[i-1].</li>
     * </ul>
     *
     * @param inputs the input feature vector
     * @return the gradient vector (bias gradient followed by weight gradients)
     */
    @Override
    public N[] calculateParameterGradients(N[] inputs) {
        N[] gradients = typeSupport.createArray(1 + arity());
        gradients[0] = typeSupport.one();
        if (arity() >= 0) System.arraycopy(inputs, 0, gradients, 1, arity());
        return gradients;
    }

    /**
     * Returns the full parameter vector of this model.
     * The first element is the bias, the rest are the weights.
     *
     * @return the parameter array
     */
    @Override
    public N[] getParameters() {
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
    public N[] getWeights() {
        return arity() > 0 ? Arrays.copyOfRange(parameters, 1, arity() + 1) : typeSupport.createArray(0);
    }

    /**
     * Replaces all weights with the given vector.
     *
     * @param weights the new weight vector
     * @throws IllegalArgumentException if the number of weights does not match {@link #arity()}
     */
    public void setWeights(N[] weights) {
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
    public N getWeight(int d) {
        return parameters[validDimension(d)];
    }

    /**
     * Sets the weight for a given dimension (1-based).
     *
     * @param d      dimension index, starting from 1
     * @param weight the new weight value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public void setWeight(int d, N weight) {
        parameters[validDimension(d)] = weight;
    }

    /**
     * Adjusts the weight for a given dimension (1-based) by adding {@code delta}.
     *
     * @param d     dimension index, starting from 1
     * @param delta the adjustment value
     * @throws IllegalArgumentException if {@code d} is out of range
     */
    public void adjustWeight(int d, N delta) {
        parameters[validDimension(d)] = typeSupport.add(parameters[d], delta);
    }

    /**
     * Returns the bias parameter.
     *
     * @return the bias value (0.0f if there are no parameters)
     */
    public N getBias() {
        return parameters.length == 0 ? typeSupport.zero() : parameters[0];
    }

    /**
     * Sets the bias parameter.
     *
     * @param bias the new bias value
     */
    public void setBias(N bias) {
        parameters[0] = bias;
    }

    /**
     * Adjusts the bias parameter by adding {@code delta}.
     *
     * @param delta the adjustment value
     */
    public void adjustBias(N delta) {
        parameters[0] = typeSupport.add(parameters[0], delta);
    }

//    public NeuralNetwork toNeuralNetwork() {
//        List<List<? extends Neuron>> neurons = new ArrayList<>();
//        List<InputNeuron> inputNeurons = new ArrayList<>();
//        for (int d = 1; d <= arity(); d++) {
//            inputNeurons.add(new InputNeuron("Input(" + (d - 1) + ")"));
//        }
//        neurons.add(inputNeurons);
//        Neuron outputNeuron = new ActivationsCachedNeuron("Output", inputNeurons, new NeuronFunction(this, Activations.identity()));
//        neurons.add(List.of(outputNeuron));
//        return new DefaultNeuralNetwork(neurons);
//    }
}
