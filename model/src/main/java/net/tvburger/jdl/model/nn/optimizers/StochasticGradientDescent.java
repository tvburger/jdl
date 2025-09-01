package net.tvburger.jdl.model.nn.optimizers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.ActivationsCachedNeuron;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.nn.NeuronVisitor;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

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
public class StochasticGradientDescent<N extends NeuralNetwork> implements Optimizer<N> {

    /**
     * Default learning rate used when none is set explicitly.
     */
    public static final float DEFAULT_LEARNING_RATE = 0.1f;

    private float learningRate = DEFAULT_LEARNING_RATE;

    /**
     * Sets the learning rate (step size) used for parameter updates.
     *
     * @param learningRate the learning rate &eta; (must be &gt; 0)
     */
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Returns the current learning rate.
     *
     * @return the learning rate &eta;
     */
    public float getLearningRate() {
        return learningRate;
    }

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
        int batchSize = trainingSet.samples().size();
        if (batchSize == 0) {
            return;
        }

        // Collect and add all the weighted votes for parameter (bias + weights) corrections
        Map<Neuron, float[]> weightedVotesForWeightCorrections = new IdentityHashMap<>();
        Map<Neuron, Float> weightedVotesForBiasCorrections = new IdentityHashMap<>();

        // For each sample n, compute error signal δ - delta - layer-by-layer (backward), accumulate gradients
        trainingSet.samples().forEach(s -> voteForParameterCorrections(s, batchSize, weightedVotesForWeightCorrections, weightedVotesForBiasCorrections, neuralNetwork, objective));

        // Apply changes
        neuralNetwork.accept(new NeuronVisitor() {
            @Override
            public void visitNeuron(NeuralNetwork neuralNetwork, Neuron neuron, int layerIndex, int neuronIndex) {
                float[] voteForWeights = weightedVotesForWeightCorrections.get(neuron);
                if (voteForWeights != null) {
                    for (int d = 1; d <= neuron.arity(); d++) {
                        neuron.adjustWeight(d, -learningRate * voteForWeights[d - 1]);
                    }
                }
                Float voteForBias = weightedVotesForBiasCorrections.get(neuron);
                if (voteForBias != null) {
                    neuron.adjustBias(-learningRate * voteForBias);
                }
                if (neuron instanceof ActivationsCachedNeuron a) {
                    a.deactivate();
                }
            }
        });
    }

    private void voteForParameterCorrections(DataSet.Sample sample, int batchSize, Map<Neuron, float[]> weightedVotesForWeightCorrections, Map<Neuron, Float> weightedVotesForBiasCorrections, N neuralNetwork, ObjectiveFunction objective) {
        Map<Neuron, Float> errorSignals = new IdentityHashMap<>();
        float[] estimated = neuralNetwork.estimate(sample.features());
        float[] lossGradients = objective.calculateGradient_dJ_da(1, estimated, sample.targetOutputs());
        for (int j = 0; j < neuralNetwork.coArity(); j++) {
            voteForOutputNodeParameterCorrections(batchSize, weightedVotesForWeightCorrections, weightedVotesForBiasCorrections, neuralNetwork, lossGradients, errorSignals, j);
        }
        for (int l = neuralNetwork.getDepth() - 1; l >= 1; l--) {
            int width = neuralNetwork.getWidth(l);
            for (int j = 0; j < width; j++) {
                voteForHiddenNodeParameterCorrections(batchSize, weightedVotesForWeightCorrections, weightedVotesForBiasCorrections, neuralNetwork, errorSignals, l, j);
            }
        }
    }

    private void voteForOutputNodeParameterCorrections(int batchSize, Map<Neuron, float[]> weightedVotesForWeightCorrections, Map<Neuron, Float> weightedVotesForBiasCorrections, N neuralNetwork, float[] lossGradients, Map<Neuron, Float> errorSignals, int j) {
        ActivationsCachedNeuron outputNode = neuralNetwork.getNeuron(neuralNetwork.getDepth(), j, ActivationsCachedNeuron.class);
        ActivationsCachedNeuron.Activation activation = outputNode.getCache().getLast();

        // determine error signal for output node
        float errorSignal = lossGradients[j] * outputNode.getActivationFunction().determineGradientForOutput(activation.output());
        errorSignals.put(outputNode, errorSignal); // track error signal for upstream nodes

        // determine bias gradient
        voteForNodeParameterCorrections(batchSize, weightedVotesForWeightCorrections, weightedVotesForBiasCorrections, errorSignal, outputNode, activation);
    }

    private void voteForHiddenNodeParameterCorrections(int batchSize, Map<Neuron, float[]> weightedVotesForWeightCorrections, Map<Neuron, Float> weightedVotesForBiasCorrections, N neuralNetwork, Map<Neuron, Float> errorSignals, int l, int j) {
        ActivationsCachedNeuron hiddenNode = neuralNetwork.getNeuron(l, j, ActivationsCachedNeuron.class);
        ActivationsCachedNeuron.Activation activation = hiddenNode.getCache().getLast();

        // determine error signal for hidden node using back propagation
        float backPropagation = 0.0f;
        Map<Neuron, Float> outs = neuralNetwork.getOutputConnections(l, j);
        for (Map.Entry<Neuron, Float> e : outs.entrySet()) {
            float connectionWeight = e.getValue();
            float errorSignalDownstream = errorSignals.getOrDefault(e.getKey(), 0.0f);
            backPropagation += errorSignalDownstream * connectionWeight;
        }
        float errorSignal = backPropagation * hiddenNode.getActivationFunction().determineGradientForOutput(activation.output());
        errorSignals.put(hiddenNode, errorSignal);

        voteForNodeParameterCorrections(batchSize, weightedVotesForWeightCorrections, weightedVotesForBiasCorrections, errorSignal, hiddenNode, activation);
    }

    private void voteForNodeParameterCorrections(int batchSize, Map<Neuron, float[]> weightedVotesForWeightCorrections, Map<Neuron, Float> weightedVotesForBiasCorrections, float errorSignal, ActivationsCachedNeuron hiddenNode, ActivationsCachedNeuron.Activation activation) {
        float[] parameterGradients_df_dp = activation.parameterGradients_df_dp();

        // determine bias gradient
        float biasGradient = errorSignal * parameterGradients_df_dp[0];
        float weightedVoteForBiasCorrection = biasGradient / batchSize;
        weightedVotesForBiasCorrections.merge(hiddenNode, weightedVoteForBiasCorrection, Float::sum); // add all up for all samples

        // determine weight gradients
        float[] inputs = activation.inputs();
        float[] gradientSumPerWeight = weightedVotesForWeightCorrections.computeIfAbsent(hiddenNode, k -> new float[inputs.length]);
        for (int d = 1; d <= inputs.length; d++) {
            float weightGradient = errorSignal * parameterGradients_df_dp[d];
            float weightedVoteForWeightCorrection = weightGradient / batchSize;
            gradientSumPerWeight[d - 1] += weightedVoteForWeightCorrection; // add the weight correction (over all samples)
        }
    }
}
