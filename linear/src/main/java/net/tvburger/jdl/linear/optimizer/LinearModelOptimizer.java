package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public interface LinearModelOptimizer<N extends Number> extends NumberTypeAgnostic<N> {

    void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet);

}
