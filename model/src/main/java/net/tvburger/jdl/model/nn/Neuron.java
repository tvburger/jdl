package net.tvburger.jdl.model.nn;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Entity;
import net.tvburger.jdl.model.scalars.AffineTransformation;
import net.tvburger.jdl.model.scalars.NeuronFunction;
import net.tvburger.jdl.model.scalars.TrainableScalarFunction;
import net.tvburger.jdl.model.scalars.activations.ActivationFunction;
import net.tvburger.jdl.model.scalars.activations.Activations;

import java.util.List;

/**
 * Represents a single artificial neuron in a neural network.
 * <p>
 * A {@code Neuron} receives inputs from other neurons, computes a weighted sum with a bias
 * (the pre-activation or "logit"), and applies an {@link ActivationFunction} to produce its output.
 * The neuron caches its most recent activation so repeated reads are cheap until explicitly
 * {@linkplain #deactivate() deactivated}.
 *
 * <h2>Computation</h2>
 * For inputs {@code x₁ … xₙ}, weights {@code w₁ … wₙ}, and bias {@code b}:
 * <pre>
 *   logit  = b + Σ (wᵢ * xᵢ)
 *   output = activationFunction.activate(logit)
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * <ul>
 *   <li>Before activation, no valid output is available.</li>
 *   <li>{@link #activate()} pulls inputs from upstream neurons, computes and caches the output.</li>
 *   <li>{@link #deactivate()} clears the activation state so the neuron can be evaluated again.</li>
 * </ul>
 *
 * <h2>Thread-safety</h2>
 * <ul>
 *   <li>{@link #activate()} and {@link #deactivate()} are {@code synchronized} to avoid races.</li>
 *   <li>Weight/bias mutation is not synchronized; coordinate externally if used concurrently.</li>
 * </ul>
 */
@DomainObject
@Entity
public class Neuron implements TrainableScalarFunction<Float> {

    public static Neuron create(String name, List<? extends Neuron> inputNodes) {
        return create(name, inputNodes, Activations.linear());
    }

    public static Neuron create(String name, List<? extends Neuron> inputNodes, ActivationFunction activationFunction) {
        return new Neuron(name, inputNodes, new NeuronFunction(AffineTransformation.create(inputNodes.size(), JavaNumberTypeSupport.FLOAT), activationFunction));
    }

    private final String name;
    private final List<? extends Neuron> inputNodes;
    private final NeuronFunction neuronFunction;
    private final Float[] inputValues;
    private float output;
    private boolean activated;

    /**
     * Constructs a new neuron with the given name, inputs, and activation function.
     * <p>
     * The parameter vector is sized as {@code inputs.size() + 1} where index {@code 0} is the bias
     * and indices {@code 1..} are the weights corresponding to the {@code inputNodes} order.
     *
     * @param name           human-readable identifier for the neuron
     * @param inputNodes     upstream input neurons in positional order; may be {@code null} for no inputs
     * @param neuronFunction the neuron function to compute a forward signal
     */
    public Neuron(String name, List<? extends Neuron> inputNodes, NeuronFunction neuronFunction) {
        this.name = name;
        this.inputNodes = inputNodes == null ? List.of() : inputNodes;
        this.inputValues = new Float[inputNodes == null ? 0 : inputNodes.size()];
        this.neuronFunction = neuronFunction;
    }

    /**
     * Returns the neuron name.
     *
     * @return the human-readable identifier
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the upstream input neurons feeding this neuron.
     *
     * @return an immutable list (possibly empty) of input neurons
     */
    public List<? extends Neuron> getInputNodes() {
        return inputNodes;
    }

    /**
     * Returns the neuron function.
     *
     * @return the neuron function
     */
    public NeuronFunction getNeuronFunction() {
        return neuronFunction;
    }

    /**
     * Returns the cached input values used during the most recent {@link #activate()}.
     *
     * @return a snapshot array of input values corresponding to {@link #getInputNodes()}
     * @throws IllegalStateException if the neuron has not been activated
     */
    public Float[] getInputValues() {
        if (!isActivated()) {
            throw new IllegalStateException("Neuron not activated!");
        }
        return inputValues;
    }

    /**
     * Computes and caches this neuron's output by:
     * <ol>
     *   <li>Reading outputs from {@link #getInputNodes()} into an internal buffer,</li>
     *   <li>Applying the {@link #getNeuronFunction()} ()}.</li>
     * </ol>
     * Subsequent calls are no-ops until {@link #deactivate()} is invoked.
     */
    public synchronized void activate() {
        if (neuronFunction == null || neuronFunction.getParameterCount() == 0) {
            activated = true;
            return;
        }
        if (isActivated()) {
            return;
        }
        for (int d = 1; d <= neuronFunction.arity(); d++) {
            inputValues[d - 1] = getInputNodes().get(d - 1).getOutput();
        }
        output = neuronFunction.estimateScalar(inputValues);
        activated = true;
    }

    /**
     * Indicates whether this neuron has a cached activation result.
     *
     * @return {@code true} if {@link #activate()} has been called and not yet {@link #deactivate()}ed; otherwise {@code false}
     */
    public boolean isActivated() {
        return activated;
    }

    /**
     * Resets the activated flag so the neuron can be re-evaluated.
     */
    public synchronized void deactivate() {
        activated = false;
    }

    /**
     * Returns the post-activation output of this neuron.
     *
     * @return the activated output value
     * @throws IllegalStateException if the neuron has not been activated
     */
    public float getOutput() {
        if (!isActivated()) {
            throw new IllegalStateException("Neuron not activated!");
        }
        return output;
    }

    /**
     * Finds the weight associated with a given source neuron.
     *
     * @param source the input neuron
     * @return the corresponding weight, or {@code null} if not found
     */
    public Float getWeight(Neuron source) {
        for (int d = 1; d <= inputNodes.size(); d++) {
            if (inputNodes.get(d - 1) == source) {
                return neuronFunction.getParameter(d);
            }
        }
        return null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return name + "{" + activated + ", " + output + "}-" + neuronFunction.toString();
    }

    @Override
    public JavaNumberTypeSupport<Float> getCurrentNumberType() {
        return JavaNumberTypeSupport.FLOAT;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int arity() {
        return neuronFunction.arity();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Float estimateScalar(Float[] inputs) {
        return neuronFunction.estimateScalar(inputs);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Float[] calculateParameterGradients(Float[] inputs) {
        return neuronFunction.calculateParameterGradients(inputs);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getParameterCount() {
        return neuronFunction.getParameterCount();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Float[] getParameters() {
        return neuronFunction.getParameters();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Float getParameter(int p) {
        return neuronFunction.getParameter(p);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setParameters(Float[] values) {
        neuronFunction.setParameters(values);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setParameter(int p, Float value) {
        neuronFunction.setParameter(p, value);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void adjustParameters(Float[] deltas) {
        neuronFunction.adjustParameters(deltas);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void adjustParameter(int p, Float delta) {
        neuronFunction.adjustParameter(p, delta);
    }
}
