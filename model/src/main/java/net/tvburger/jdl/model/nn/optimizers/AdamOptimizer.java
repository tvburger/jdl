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

/**
 * Adam optimizer (Kingma &amp; Ba, 2014).
 * Mirrors GradientDescent's training loop: we accumulate per-batch gradients,
 * then apply Adam updates with bias correction.
 */
@Strategy(role = Strategy.Role.CONCRETE)
public class AdamOptimizer<N extends NeuralNetwork> implements Optimizer<N> {

    public static final float DEFAULT_LEARNING_RATE = 1e-3f;   // alpha
    public static final float DEFAULT_BETA1 = 0.9f;
    public static final float DEFAULT_BETA2 = 0.999f;
    public static final float DEFAULT_EPSILON = 1e-8f;

    private float learningRate = DEFAULT_LEARNING_RATE;
    private float beta1 = DEFAULT_BETA1;
    private float beta2 = DEFAULT_BETA2;
    private float epsilon = DEFAULT_EPSILON;

    // Adam state: first (m) and second (v) moments for weights and biases
    private final Map<Neuron, float[]> mW = new IdentityHashMap<>();
    private final Map<Neuron, float[]> vW = new IdentityHashMap<>();
    private final Map<Neuron, Float> mB = new IdentityHashMap<>();
    private final Map<Neuron, Float> vB = new IdentityHashMap<>();

    // Global step for bias correction
    private long t = 0L;

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getBeta1() {
        return beta1;
    }

    public void setBeta1(float beta1) {
        this.beta1 = beta1;
    }

    public float getBeta2() {
        return beta2;
    }

    public void setBeta2(float beta2) {
        this.beta2 = beta2;
    }

    public float getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public void optimize(N neuralNetwork, DataSet trainingSet, ObjectiveFunction objective) {
        final int sampleCount = trainingSet.samples().size();
        if (sampleCount == 0) return;

        final int depth = neuralNetwork.getDepth();

        // ---- Accumulate per-batch gradients (same as your GD) ----
        final Map<Neuron, float[]> dWsum = new IdentityHashMap<>();
        final Map<Neuron, Float> dBsum = new IdentityHashMap<>();

        for (int n = 0; n < sampleCount; n++) {
            final Map<Neuron, Float> deltaN = new IdentityHashMap<>();

            // Output layer deltas
            int outWidth = neuralNetwork.coArity();
            DataSet.Sample sample = trainingSet.samples().get(n);
            float[] estimated = neuralNetwork.estimate(sample.features());
            float[] lossGradients = objective.determineGradients(estimated, sample.targetOutputs());

            for (int j = 0; j < outWidth; j++) {
                ActivationsCachedNeuron out = neuralNetwork.getNeuron(depth, j, ActivationsCachedNeuron.class);
                ActivationsCachedNeuron.Activation activation = out.getCache().get(n);

                float delta = lossGradients[j] * activation.gradient();
                deltaN.put(out, delta);

                float[] inputs = activation.inputs();
                float[] gsum = dWsum.computeIfAbsent(out, k -> new float[inputs.length]);
                for (int i = 0; i < inputs.length; i++) gsum[i] += delta * inputs[i];
                dBsum.merge(out, delta, Float::sum);
            }

            // Hidden layers deltas (backward)
            for (int layer = depth - 1; layer >= 1; layer--) {
                int width = neuralNetwork.getWidth(layer);
                for (int j = 0; j < width; j++) {
                    ActivationsCachedNeuron neuron = neuralNetwork.getNeuron(layer, j, ActivationsCachedNeuron.class);

                    float back = 0f;
                    Map<Neuron, Float> outs = neuralNetwork.getOutputConnections(layer, j);
                    for (Map.Entry<Neuron, Float> e : outs.entrySet()) {
                        Float deltaNext = deltaN.get(e.getKey());
                        if (deltaNext != null) back += deltaNext * e.getValue();
                    }

                    ActivationsCachedNeuron.Activation activation = neuron.getCache().get(n);
                    float delta = back * activation.gradient();
                    deltaN.put(neuron, delta);

                    float[] inputs = activation.inputs();
                    float[] gsum = dWsum.computeIfAbsent(neuron, k -> new float[inputs.length]);
                    for (int i = 0; i < inputs.length; i++) gsum[i] += delta * inputs[i];
                    dBsum.merge(neuron, delta, Float::sum);
                }
            }
        }

        // ---- Adam update on averaged gradients ----
        t += 1L; // step
        final float invN = 1f / sampleCount;

        // Precompute bias correction factors
        final double b1t = Math.pow(beta1, t);
        final double b2t = Math.pow(beta2, t);
        final float biasCorr1 = (float) (1.0 / (1.0 - b1t));
        final float biasCorr2 = (float) (1.0 / (1.0 - b2t));

        for (int layer = 1; layer <= depth; layer++) {
            int width = neuralNetwork.getWidth(layer);
            for (int j = 0; j < width; j++) {
                ActivationsCachedNeuron neuron = neuralNetwork.getNeuron(layer, j, ActivationsCachedNeuron.class);

                // Weights
                float[] w = neuron.getWeights();
                float[] gWsum = dWsum.get(neuron);

                if (gWsum != null) {
                    // Ensure moment buffers exist and match shape
                    float[] mWi = mW.computeIfAbsent(neuron, k -> new float[w.length]);
                    float[] vWi = vW.computeIfAbsent(neuron, k -> new float[w.length]);
                    if (mWi.length != w.length) {
                        mWi = new float[w.length];
                        vWi = new float[w.length];
                        mW.put(neuron, mWi);
                        vW.put(neuron, vWi);
                    }

                    for (int i = 0; i < w.length; i++) {
                        // Average gradient over batch
                        float g = invN * gWsum[i];

                        // Adam moment updates
                        mWi[i] = beta1 * mWi[i] + (1 - beta1) * g;
                        vWi[i] = beta2 * vWi[i] + (1 - beta2) * (g * g);

                        // Bias-corrected moments
                        float mHat = mWi[i] * biasCorr1;
                        float vHat = vWi[i] * biasCorr2;

                        // Parameter update
                        w[i] -= learningRate * (mHat / (float) (Math.sqrt(vHat) + (double) epsilon));
                    }
                }

                // Bias
                Float gBsum = dBsum.get(neuron);
                if (gBsum != null) {
                    float g = invN * gBsum;

                    float mBi = mB.getOrDefault(neuron, 0f);
                    float vBi = vB.getOrDefault(neuron, 0f);

                    mBi = beta1 * mBi + (1 - beta1) * g;
                    vBi = beta2 * vBi + (1 - beta2) * (g * g);

                    float mHat = mBi * biasCorr1;
                    float vHat = vBi * biasCorr2;

                    float b = neuron.getBias();
                    b -= learningRate * (mHat / (float) (Math.sqrt(vHat) + (double) epsilon));
                    neuron.setBias(b);

                    mB.put(neuron, mBi);
                    vB.put(neuron, vBi);
                }

                // Clear per-batch caches like in GD
                neuron.clearCache();
            }
        }
    }
}
