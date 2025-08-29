package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.MarkerInterface;

/**
 * Marker interface for all types of loss functions.
 *
 * <p>
 * A {@code LossFunction} defines the contract for calculating error or
 * "loss" between estimated (predicted) values and target (expected) values
 * in the context of optimization, training, or evaluation.
 */
@DomainObject
@MarkerInterface
public interface LossFunction {
}
