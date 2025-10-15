package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

public class LearningRateWarmup<N extends Number> implements GradientDescentOptimizer.Interceptor<N> {

    private final int warmupEpochs;

    public LearningRateWarmup(int warmupEpochs) {
        this.warmupEpochs = warmupEpochs;
    }

    @Override
    public Vector<N> interceptGradients(int epoch, LinearBasisFunctionModel<N> model, DataSet<N> trainSet, GradientDescentOptimizer<N> optimizer, Vector<N> gradients) {
        return epoch > warmupEpochs ? gradients : warmUpGradient(epoch, gradients, model.getCurrentNumberType());
    }

    private Vector<N> warmUpGradient(int epoch, Vector<N> gradients, JavaNumberTypeSupport<N> numberType) {
        N epochNumber = numberType.valueOf(epoch);
        N warmupEpochsNumber = numberType.valueOf(this.warmupEpochs);
        N scale = numberType.divide(epochNumber, warmupEpochsNumber);
        return gradients.multiply(scale);
    }

}
