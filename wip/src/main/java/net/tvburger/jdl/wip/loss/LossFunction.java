package net.tvburger.jdl.wip.loss;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.nn.NeuralNetwork;

/**
 * Defines a loss (or cost) function used to measure the discrepancy
 * between the expected outputs (targets) of a dataset and the predictions
 * made by a {@link NeuralNetwork}.
 * <p>
 * A loss function serves two key purposes:
 * <ul>
 *   <li>Quantify model performance by producing a scalar loss value
 *       (e.g., mean squared error, cross-entropy).</li>
 *   <li>Provide gradients with respect to the outputs, enabling
 *       learning algorithms (e.g., gradient descent) to update
 *       model parameters.</li>
 * </ul>
 *
 * <h2>Common examples</h2>
 * <ul>
 *   <li><b>Mean Squared Error (MSE):</b> {@code (y - ŷ)²}</li>
 *   <li><b>Cross-Entropy:</b> {@code -Σ y · log(ŷ)}</li>
 * </ul>
 */
public interface LossFunction<E extends EstimationFunction> {

    float calculateLoss(float error);

    default float calculateLoss(DataSet dataSet, E estimationFunction) {
        float[] outputErrors = calculateOutputErrors(dataSet, estimationFunction);
        float totalLoss = 0.0f;
        for (float outputError : outputErrors) {
            totalLoss += calculateLoss(outputError);
        }
        return totalLoss / outputErrors.length;
    }

    default float calculateAggregateError(DataSet dataSet, E estimationFunction) {
        float[] outputErrors = calculateOutputErrors(dataSet, estimationFunction);
        float totalError = 0.0f;
        for (float outputError : outputErrors) {
            totalError += outputError;
        }
        return totalError / outputErrors.length;
    }

    float[] calculateOutputErrors(DataSet dataSet, E estimationFunction);

    float[] determineGradients(DataSet dataSet, E estimationFunction);

}
