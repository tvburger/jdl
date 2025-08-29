package net.tvburger.jdl.model.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
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
@DomainObject
@Strategy(role = Strategy.Role.INTERFACE)
public interface LossFunction {

    float calculateLoss(DataSet dataSet, EstimationFunction estimationFunction);

    float[] calculateOutputErrors(DataSet dataSet, EstimationFunction estimationFunction);

    float[] determineGradients(DataSet dataSet, EstimationFunction estimationFunction);

}
