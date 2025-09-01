package net.tvburger.jdl.model.nn.training.optimizers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.ActivationsCachedNeuron;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.NeuronVisitor;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.optimizer.LearningRateConfigurable;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Stochastic Gradient Descent (SGD) optimizer with per-sample backpropagation and
 * batch accumulation of gradients.
 * <p>
 * This implementation:
 * <ul>
 *   <li>Performs a forward pass for each sample, computes error signals (δ), and
 *       accumulates weight/bias gradients over the given {@link DataSet}.</li>
 *   <li>Applies a single parameter update after processing the full set (i.e., a mini-batch/batch step).</li>
 *   <li>Relies on {@link ActivationsCachedNeuron} to access cached forward-pass values needed for backprop.</li>
 *   <li>Deactivates neuron activation caches after parameters are updated.</li>
 * </ul>
 *
 * <p><strong>Batch scaling convention:</strong> This class, as written, assumes the provided
 * {@link ObjectiveFunction#calculateGradient_dJ_da(int, float[], float[])} returns
 * gradients already scaled by the batch size (i.e., mean over the batch). Consequently,
 * parameter updates do not divide by the batch size again on apply.</p>
 *
 * @param <N> a {@link NeuralNetwork} type optimized by this optimizer
 */
@Strategy(Strategy.Role.CONCRETE)
public class StochasticGradientDescent<N extends NeuralNetwork> implements Optimizer<N>, LearningRateConfigurable {

    /**
     * Default learning rate used when none is set explicitly.
     */
    public static final float DEFAULT_LEARNING_RATE = 0.1f;

    private final Map<String, Object> hyperparameters = new HashMap<>() {{
        put(HP_LEARNING_RATE, DEFAULT_LEARNING_RATE);
    }};

    /**
     * Runs one optimization step over the provided training set.
     * <p>
     * For each sample, this method:
     * <ol>
     *   <li>Performs a forward pass to get network outputs.</li>
     *   <li>Computes the loss gradient w.r.t. activations from the {@link ObjectiveFunction}.</li>
     *   <li>Backpropagates error signals (δ) layer-by-layer to accumulate gradients for all weights and biases.</li>
     * </ol>
     * After all samples are processed, it applies the accumulated gradients to the network parameters
     * using the configured learning rate, and deactivates activation caches.
     *
     * <p><strong>Note:</strong> This implementation assumes {@code objective.calculateGradient_dJ_da}
     * returns gradients that are already averaged over the batch size passed in. Therefore the
     * final parameter updates do not perform an additional division by the batch size.</p>
     *
     * @param neuralNetwork the network to optimize
     * @param trainingSet   the data used for one optimization step (treated here as a batch)
     * @param objective     the objective/loss function providing gradients w.r.t. output activations
     */
    @Override
    public void optimize(N neuralNetwork, DataSet trainingSet, ObjectiveFunction objective) {
        int batchSize = trainingSet.size();
        if (batchSize == 0) {
            return;
        }

        // Collect and add all the weighted votes for parameter (bias + weights) corrections
        Map<Neuron, float[]> accumulatedWeightedVotesForParameterCorrections = new IdentityHashMap<>();

        // For each sample n, compute error signal δ - delta - layer-by-layer (backward), accumulate gradients
        trainingSet.forEach(s -> voteForParameterCorrections(s, batchSize, accumulatedWeightedVotesForParameterCorrections, neuralNetwork, objective));

        // Apply changes
        neuralNetwork.accept(new NeuronVisitor() {
            @Override
            public void visitNeuron(NeuralNetwork neuralNetwork, Neuron neuron, int layerIndex, int neuronIndex) {
                float[] parameterCorrections = accumulatedWeightedVotesForParameterCorrections.get(neuron);
                if (parameterCorrections != null) {
                    for (int p = 0; p < parameterCorrections.length; p++) {
                        neuron.adjustParameters(p, getLearningRate() * -1 * parameterCorrections[p]);
                    }
                }
            }
        });
    }

    private void voteForParameterCorrections(DataSet.Sample sample, int batchSize, Map<Neuron, float[]> accumulatedWeightedVotesForParameterCorrections, N neuralNetwork, ObjectiveFunction objective) {
        Map<Neuron, Float> errorSignals = new IdentityHashMap<>();
        float[] estimated = neuralNetwork.estimate(sample.features());
        float[] lossGradients = objective.calculateGradient_dJ_da(1, estimated, sample.targetOutputs());
        for (int j = 0; j < neuralNetwork.coArity(); j++) {
            voteForOutputNodeParameterCorrections(batchSize, accumulatedWeightedVotesForParameterCorrections, neuralNetwork, lossGradients, errorSignals, j);
        }
        for (int l = neuralNetwork.getDepth() - 1; l >= 1; l--) {
            int width = neuralNetwork.getWidth(l);
            for (int j = 0; j < width; j++) {
                voteForHiddenNodeParameterCorrections(batchSize, accumulatedWeightedVotesForParameterCorrections, neuralNetwork, errorSignals, l, j);
            }
        }
    }

    private void voteForOutputNodeParameterCorrections(int batchSize, Map<Neuron, float[]> accumulatedWeightedVotesForParameterCorrections, N neuralNetwork, float[] lossGradients, Map<Neuron, Float> errorSignals, int j) {
        ActivationsCachedNeuron outputNode = neuralNetwork.getNeuron(neuralNetwork.getDepth(), j, ActivationsCachedNeuron.class);
        ActivationsCachedNeuron.Activation activation = outputNode.getCache().removeLast();

        // determine error signal for output node
        float errorSignal = lossGradients[j] * outputNode.getActivationFunction().determineGradientForOutput(activation.output());
        errorSignals.put(outputNode, errorSignal); // track error signal for upstream nodes

        // cast vote
        voteForNodeParameterCorrections(batchSize, accumulatedWeightedVotesForParameterCorrections, errorSignal, outputNode, activation);
    }

    private void voteForHiddenNodeParameterCorrections(int batchSize, Map<Neuron, float[]> accumulatedWeightedVotesForParameterCorrections, N neuralNetwork, Map<Neuron, Float> errorSignals, int l, int j) {
        ActivationsCachedNeuron hiddenNode = neuralNetwork.getNeuron(l, j, ActivationsCachedNeuron.class);
        ActivationsCachedNeuron.Activation activation = hiddenNode.getCache().removeLast();

        // determine error signal for hidden node using back propagation
        float backPropagation = calculateBackPropagation(neuralNetwork, errorSignals, l, j);
        float errorSignal = backPropagation * hiddenNode.getActivationFunction().determineGradientForOutput(activation.output());
        errorSignals.put(hiddenNode, errorSignal);

        // cast vote
        voteForNodeParameterCorrections(batchSize, accumulatedWeightedVotesForParameterCorrections, errorSignal, hiddenNode, activation);
    }

    private static <N extends NeuralNetwork> float calculateBackPropagation(N neuralNetwork, Map<Neuron, Float> errorSignals, int l, int j) {
        float backPropagation = 0.0f;
        Map<Neuron, Float> downstreamNodes = neuralNetwork.getOutputConnections(l, j);
        for (Map.Entry<Neuron, Float> e : downstreamNodes.entrySet()) {
            float downstreamWeight = e.getValue();
            float downstreamErrorSignal = errorSignals.getOrDefault(e.getKey(), 0.0f);
            backPropagation += downstreamErrorSignal * downstreamWeight;
        }
        return backPropagation;
    }

    private void voteForNodeParameterCorrections(int batchSize, Map<Neuron, float[]> accumulatedWeightedVotesForParameterCorrections, float errorSignal, ActivationsCachedNeuron neuron, ActivationsCachedNeuron.Activation activation) {
        float[] parameterGradients_df_dp = activation.parameterGradients_df_dp();
        int parameterCount = neuron.getParameterCount();
        float[] votesForNeuron = accumulatedWeightedVotesForParameterCorrections.computeIfAbsent(neuron, k -> new float[parameterCount]);
        for (int p = 0; p < parameterCount; p++) {
            float parameterCorrection = errorSignal * parameterGradients_df_dp[p];
            float weightedVote = parameterCorrection / batchSize;
            votesForNeuron[p] += weightedVote;
        }
    }

    /**
     * {@inheritDoc
     */
    @Override
    public Map<String, Object> getHyperparameters() {
        return hyperparameters;
    }

    /**
     * {@inheritDoc
     */
    @Override
    public void setHyperparameter(String name, Object value) {
        hyperparameters.put(name, value);
    }
}
