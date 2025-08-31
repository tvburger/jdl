package net.tvburger.jdl.model.nn.optimizers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.ActivationsCachedNeuron;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

import java.util.IdentityHashMap;
import java.util.Map;

@Strategy(Strategy.Role.CONCRETE)
public class StochasticGradientDescent<N extends NeuralNetwork> implements Optimizer<N> {

    public static final float DEFAULT_LEARNING_RATE = 0.1f;

    private float learningRate = DEFAULT_LEARNING_RATE;

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }

    @Override
    public void optimize(N neuralNetwork, DataSet trainingSet, ObjectiveFunction objective) {
        int batchSize = trainingSet.samples().size();
        if (batchSize == 0) {
            return;
        }
        int depth = neuralNetwork.getDepth();

        // Allocate accumulators for gradients (sum over batch)
        final Map<Neuron, float[]> dWsum = new IdentityHashMap<>();
        final Map<Neuron, Float> dBsum = new IdentityHashMap<>();

        // For each sample n, compute error signal δ - delta - layer-by-layer (backward), accumulate gradients
        for (int i = 0; i < batchSize; i++) {

            // 3a) Per-sample δ storage for this backward sweep
            final Map<Neuron, Float> deltaN = new IdentityHashMap<>();

            // 3b) OUTPUT LAYER δ
            int outWidth = neuralNetwork.coArity();
            DataSet.Sample sample = trainingSet.samples().get(i);
            float[] estimated = neuralNetwork.estimate(sample.features());
            float[] lossGradients = objective.calculateGradient_dJ_da(trainingSet.samples().size(), estimated, sample.targetOutputs());

            // we calculate the δ for each output node
            for (int j = 0; j < outWidth; j++) {
                ActivationsCachedNeuron out = neuralNetwork.getNeuron(depth, j, ActivationsCachedNeuron.class);

                // Fetch cached data for this sample
                ActivationsCachedNeuron.Activation activation = out.getCache().get(i);
                float delta = lossGradients[j] * activation.gradient();
                deltaN.put(out, delta);

                // Accumulate grads for this neuron: dW += δ * inputs^{(n)}, dB += δ
                float[] inputs = activation.inputs();  // a_i^{(n)} (inputs into this neuron)
                float[] gradientSumPerWeight = dWsum.computeIfAbsent(out, k -> new float[inputs.length]);
                for (int d = 0; d < inputs.length; d++) gradientSumPerWeight[d] += delta * inputs[d];
                dBsum.merge(out, delta, Float::sum);
            }

            // 3c) HIDDEN LAYERS δ: for l = L-1 .. 1
            for (int l = depth - 1; l >= 1; l--) {
                int width = neuralNetwork.getWidth(l);
                for (int j = 0; j < width; j++) {
                    ActivationsCachedNeuron neuron = neuralNetwork.getNeuron(l, j, ActivationsCachedNeuron.class);

                    // sum_k w_jk * δ_k^{(n)} from the NEXT layer (already computed this sample)
                    float back = 0f;
                    Map<Neuron, Float> outs = neuralNetwork.getOutputConnections(l, j);
                    for (Map.Entry<Neuron, Float> e : outs.entrySet()) {
                        Float deltaNext = deltaN.get(e.getKey());
                        if (deltaNext != null) back += deltaNext * e.getValue();
                    }

                    // multiply by local activation derivative f'(z^{(n)})
                    ActivationsCachedNeuron.Activation activation = neuron.getCache().get(i);
                    float fprime = activation.gradient();
                    float delta = back * fprime;
                    deltaN.put(neuron, delta);

                    // Accumulate grads for this neuron
                    float[] inputs = activation.inputs(); // a_i^{(n)}
                    float[] gradientSumPerWeight = dWsum.computeIfAbsent(neuron, k -> new float[inputs.length]);
                    for (int d = 0; d < inputs.length; d++) gradientSumPerWeight[d] += delta * inputs[d];
                    dBsum.merge(neuron, delta, Float::sum);
                }
            }
        }

        // Apply averaged gradients: w := w - η * (1/N) * dWsum ; b := b - η * (1/N) * dBsum
        final float invN = 1f / batchSize;
        for (int l = 1; l <= depth; l++) {
            int width = neuralNetwork.getWidth(l);
            for (int j = 0; j < width; j++) {
                ActivationsCachedNeuron neuron = neuralNetwork.getNeuron(l, j, ActivationsCachedNeuron.class);

                // We may have "dead" units in ReLU → if never used, skip
                float[] w = neuron.getWeights();
                float[] gw = dWsum.get(neuron);
                if (gw != null) {
                    for (int d = 0; d < w.length; d++) {
                        w[d] -= learningRate * invN * gw[d];
                    }
                }
                Float gb = dBsum.get(neuron);
                if (gb != null) {
                    neuron.setBias(neuron.getBias() - learningRate * invN * gb);
                }

                // clear per-batch caches
                neuron.clearCache();
            }
        }
    }
}
