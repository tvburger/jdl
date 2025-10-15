package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

import java.util.function.Consumer;

public interface IterativeOptimizer<N extends Number> {

    @FunctionalInterface
    interface EpochCompletionListener<N extends Number> {

        void epochCompleted(int epoch, LinearBasisFunctionModel<N> regression, DataSet<N> trainSet, IterativeOptimizer<N> optimizer);

    }

    static <N extends Number> EpochCompletionListener<N> initializer(Consumer<LinearBasisFunctionModel<N>> initializer) {
        return (epoch, regression, trainSet, optimizer) -> {
            if (epoch == 1) {
                initializer.accept(regression);
            }
        };
    }

    static <N extends Number> EpochCompletionListener<N> initializer(Consumer<LinearBasisFunctionModel<N>> initializer, EpochCompletionListener<N> listener) {
        return (epoch, regression, trainSet, optimizer) -> {
            if (epoch == 1) {
                initializer.accept(regression);
            }
            listener.epochCompleted(epoch, regression, trainSet, optimizer);
        };
    }

    static <N extends Number> EpochCompletionListener<N> sample(int n, EpochCompletionListener<N> listener) {
        return (epoch, regression, trainSet, optimizer) -> {
            boolean call;
            int totalEpochs = optimizer.getEpochs();
            if (totalEpochs < n || totalEpochs == epoch || epoch == 1) {
                call = true;
            } else {
                call = (epoch % (totalEpochs / n)) == 0;
            }
            if (call) {
                listener.epochCompleted(epoch, regression, trainSet, optimizer);
            }
        };
    }

    int getEpochs();

    void setEpochs(int epochs);

    float getLearningRate();

    void setLearningRate(float learningRate);

    EpochCompletionListener<N> getEpochCompletionListener();

    void setEpochCompletionListener(EpochCompletionListener<N> epochCompletionListener);

}
