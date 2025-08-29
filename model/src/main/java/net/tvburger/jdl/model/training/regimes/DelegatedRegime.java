package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

@Decorator
public abstract class DelegatedRegime implements Regime {

    protected final Regime regime;

    protected DelegatedRegime(Regime regime) {
        this.regime = regime != null ? regime : new Regime() {
            @Override
            public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
                optimizer.optimize(estimationFunction, trainingSet.compatible(estimationFunction), objective);
            }
        };
    }

    public Regime getDelegatedRegime() {
        return regime;
    }

    public BatchRegime batch() {
        return new BatchRegime(this);
    }

    public OnlineRegime online() {
        return new OnlineRegime(this);
    }

    public EpochRegime epoch(int epochs) {
        return new EpochRegime(this, epochs);
    }

    public MiniBatchRegime miniBatch(int batchSize) {
        return new MiniBatchRegime(this, batchSize);
    }
}
