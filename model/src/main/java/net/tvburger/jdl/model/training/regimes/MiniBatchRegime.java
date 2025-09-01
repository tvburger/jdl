package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

@Strategy(Strategy.Role.CONCRETE)
public final class MiniBatchRegime extends DelegatedRegime implements BatchSizeConfigurable {

    /**
     * Creates a new mini-batch trainer with the given mini-batch size.
     *
     * @param batchSize the number of samples per mini-batch (must be > 0)
     */
    public MiniBatchRegime(int batchSize) {
        super(new BatchRegime());
        setHyperparameter(HP_BATCH_SIZE, batchSize);
    }

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        int offset = 0;
        int trainingSetSize = trainingSet.size();
        do {
            int newOffset = Math.min(trainingSetSize, offset + getBatchSize());
            regime.train(estimationFunction, trainingSet.subset(offset, newOffset), objective, optimizer);
            offset = newOffset;
        } while (offset >= trainingSet.size());
    }
}
