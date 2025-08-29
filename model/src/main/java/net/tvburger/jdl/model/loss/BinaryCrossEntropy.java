package net.tvburger.jdl.model.loss;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

import java.util.List;

/**
 * Binary Cross-Entropy (BCE) loss function.
 * <p>
 * This loss function is used for binary classification tasks
 * where each output neuron predicts the probability of a binary label
 * (0 or 1) using a {@code sigmoid} activation function.
 * It computes the cross-entropy between the predicted probability
 * and the target label.
 * </p>
 *
 * <h3>Definition</h3>
 * For a single output:
 * <pre>
 *     E = - [ y * log(a) + (1 - y) * log(1 - a) ]
 * </pre>
 * where:
 * <ul>
 *   <li>{@code y} is the true label in {0,1}</li>
 *   <li>{@code a} is the predicted probability (sigmoid output)</li>
 * </ul>
 *
 * <h3>Gradients</h3>
 * The derivative of the BCE loss with respect to the output activation {@code a} is:
 * <pre>
 *     dE/da = -( y / a - (1 - y) / (1 - a) )
 * </pre>
 * When combined with the derivative of the sigmoid activation {@code f'(z) = a(1 - a)},
 * this simplifies to the familiar error signal:
 * <pre>
 *     δ = a - y
 * </pre>
 *
 * <h3>Usage</h3>
 * <ul>
 *   <li>Output layer: one sigmoid unit per target</li>
 *   <li>Loss function: {@code BinaryCrossEntropy}</li>
 *   <li>Training: use the error signals δ = a - y as the starting point for backpropagation</li>
 * </ul>
 *
 * <h3>Numerical stability</h3>
 * To avoid {@code log(0)} or division by zero, a small epsilon (e.g. 1e-7)
 * should be added when clamping predicted probabilities {@code a} into (0,1).
 *
 * @see LossFunction
 */
@Strategy(role = Strategy.Role.CONCRETE)
public class BinaryCrossEntropy implements LossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public float calculateLoss(DataSet dataSet, EstimationFunction estimationFunction) {
        List<DataSet.Sample> samples = dataSet.samples();
        if (samples.isEmpty()) return 0f;

        float total = 0f;
        int totalOutputs = 0;

        for (DataSet.Sample s : samples) {
            float[] x = s.features();
            float[] y = s.targetOutputs();
            float[] a = estimationFunction.estimate(x); // predicted activations in [0,1]

            for (int k = 0; k < y.length; k++) {
                float ak = clamp01(a[k]);
                float yk = y[k];
                // Binary cross-entropy for this output
                total += -(yk * (float) Math.log(ak + Floats.EPSILON)
                        + (1f - yk) * (float) Math.log(1f - ak + Floats.EPSILON));
                totalOutputs++;
            }
        }

        return totalOutputs == 0 ? 0f : total / totalOutputs;
    }

    /**
     * {@inheritDoc}
     */

    @Override
    public float[] calculateOutputErrors(DataSet dataSet, EstimationFunction estimationFunction) {
        return determineGradients(dataSet, estimationFunction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] determineGradients(DataSet dataSet, EstimationFunction estimationFunction) {
        List<DataSet.Sample> samples = dataSet.samples();
        if (samples.isEmpty()) return new float[0];

        // Infer output dimension from first sample
        float[] sum = new float[estimationFunction.coArity()];
        int count = 0;

        for (DataSet.Sample s : samples) {
            float[] x = s.features();
            float[] y = s.targetOutputs();
            float[] a = estimationFunction.estimate(x);

            // safety if some sample has different length
            int len = Math.min(sum.length, Math.min(y.length, a.length));

            for (int k = 0; k < len; k++) {
                float ak = clamp01(a[k]);
                float yk = y[k];
                // dE/da for binary cross-entropy
                float dE_da = -(yk / (ak + Floats.EPSILON) - (1f - yk) / (1f - ak + Floats.EPSILON));
                sum[k] += dE_da;
            }
            count++;
        }

        // Mean over the batch
        if (count > 0) {
            for (int k = 0; k < sum.length; k++) sum[k] /= count;
        }
        return sum;
    }

    private static float clamp01(float f) {
        if (f < Floats.EPSILON) return Floats.EPSILON;
        if (f > 1f - Floats.EPSILON) return 1f - Floats.EPSILON;
        return f;
    }

}
