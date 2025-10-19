package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

public interface LinearModelOptimizer<N extends Number> extends NumberTypeAgnostic<N>, Optimizer<LinearBasisFunctionModel<N>, N> {

    void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet);

    @Override
    default void optimize(LinearBasisFunctionModel<N> estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, int step) {
        setOptimalWeights(estimationFunction, trainingSet);
    }

}
