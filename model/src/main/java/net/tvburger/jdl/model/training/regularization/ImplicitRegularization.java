package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.model.DataSet;

/**
 * Implicit regularization comes from how the training is performed, not from an explicit penalty in the loss
 *
 * @param <N> Number type
 */
public interface ImplicitRegularization<N extends Number> extends Regularization<N> {

    default N afterUpdateConstrain(N parameter) {
        return parameter;
    }

    default DataSet<N> onBatch(DataSet<N> trainingSet) {
        return trainingSet;
    }

    default Array<N> onActivations(int layer, Array<N> activations) {
        return activations;
    }

    default DataSet<N> onTargets(DataSet<N> trainingSet) {
        return trainingSet;
    }

    default boolean shouldStop(int epoch, N trainingLoss) {
        return false;
    }

}
