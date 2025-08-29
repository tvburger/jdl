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

@Strategy(role = Strategy.Role.CONCRETE)
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
        int sampleCount = trainingSet.samples().size();
        if (sampleCount == 0) {
            return;
        }
        int depth = neuralNetwork.getDepth();

        // Allocate accumulators for gradients (sum over batch)
        final Map<Neuron, float[]> dWsum = new IdentityHashMap<>();
        final Map<Neuron, Float> dBsum = new IdentityHashMap<>();

        // For each sample n, compute error signal δ - delta - layer-by-layer (backward), accumulate gradients
        for (int n = 0; n < sampleCount; n++) {

            // 3a) Per-sample δ storage for this backward sweep
            final Map<Neuron, Float> deltaN = new IdentityHashMap<>();

            // 3b) OUTPUT LAYER δ
            int outWidth = neuralNetwork.coArity();
            DataSet.Sample sample = trainingSet.samples().get(n);
            float[] estimated = neuralNetwork.estimate(sample.features());
            float[] lossGradients = objective.determineGradients(estimated, sample.targetOutputs());

            // we calculate the δ for each output node
            for (int j = 0; j < outWidth; j++) {
                ActivationsCachedNeuron out = neuralNetwork.getNeuron(depth, j, ActivationsCachedNeuron.class);

                // Fetch cached data for this sample
                ActivationsCachedNeuron.Activation activation = out.getCache().get(n);
                float delta = lossGradients[j] * activation.gradient();
                deltaN.put(out, delta);

                // Accumulate grads for this neuron: dW += δ * inputs^{(n)}, dB += δ
                float[] inputs = activation.inputs();  // a_i^{(n)} (inputs into this neuron)
                float[] gradientSumPerWeight = dWsum.computeIfAbsent(out, k -> new float[inputs.length]);
                for (int i = 0; i < inputs.length; i++) gradientSumPerWeight[i] += delta * inputs[i];
                dBsum.merge(out, delta, Float::sum);
            }

            // 3c) HIDDEN LAYERS δ: for l = L-1 .. 1
            for (int layer = depth - 1; layer >= 1; layer--) {
                int width = neuralNetwork.getWidth(layer);
                for (int j = 0; j < width; j++) {
                    ActivationsCachedNeuron neuron = neuralNetwork.getNeuron(layer, j, ActivationsCachedNeuron.class);

                    // sum_k w_jk * δ_k^{(n)} from the NEXT layer (already computed this sample)
                    float back = 0f;
                    Map<Neuron, Float> outs = neuralNetwork.getOutputConnections(layer, j);
                    for (Map.Entry<Neuron, Float> e : outs.entrySet()) {
                        Float deltaNext = deltaN.get(e.getKey());
                        if (deltaNext != null) back += deltaNext * e.getValue();
                    }

                    // multiply by local activation derivative f'(z^{(n)})
                    ActivationsCachedNeuron.Activation activation = neuron.getCache().get(n);
                    float fprime = activation.gradient();
                    float delta = back * fprime;
                    deltaN.put(neuron, delta);

                    // Accumulate grads for this neuron
                    float[] inputs = activation.inputs(); // a_i^{(n)}
                    float[] gradientSumPerWeight = dWsum.computeIfAbsent(neuron, k -> new float[inputs.length]);
                    for (int i = 0; i < inputs.length; i++) gradientSumPerWeight[i] += delta * inputs[i];
                    dBsum.merge(neuron, delta, Float::sum);
                }
            }
        }

        // Apply averaged gradients: w := w - η * (1/N) * dWsum ; b := b - η * (1/N) * dBsum
        final float invN = 1f / sampleCount;
        for (int layer = 1; layer <= depth; layer++) {
            int width = neuralNetwork.getWidth(layer);
            for (int j = 0; j < width; j++) {
                ActivationsCachedNeuron neuron = neuralNetwork.getNeuron(layer, j, ActivationsCachedNeuron.class);

                // We may have "dead" units in ReLU → if never used, skip
                float[] w = neuron.getWeights();
                float[] gw = dWsum.get(neuron);
                if (gw != null) {
                    for (int i = 0; i < w.length; i++) {
                        w[i] -= learningRate * invN * gw[i];
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
