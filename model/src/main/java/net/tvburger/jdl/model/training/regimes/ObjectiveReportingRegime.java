package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

import java.util.List;

public class ObjectiveReportingRegime extends DelegatedRegime {

    private Float improvement;
    private Float previousLoss;
    private int iteration;

    public ObjectiveReportingRegime(Regime regime) {
        super(regime);
    }

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        if (iteration == 0 && objective != null) {
            List<Pair<float[], float[]>> batch = trainingSet.samples().stream().map(s -> Pair.of(estimationFunction.estimate(s.features()), s.targetOutputs())).toList();
            float aggregatedLoss = objective.calculateAggregatedLoss(batch);
            System.out.printf("[Epoch %4d] Aggregated loss = %.4f (baseline)\n", iteration, aggregatedLoss);
            previousLoss = aggregatedLoss;
        }
        iteration++;
        regime.train(estimationFunction, trainingSet, objective, optimizer);
        if (objective != null) {
            List<Pair<float[], float[]>> batch = trainingSet.samples().stream().map(s -> Pair.of(estimationFunction.estimate(s.features()), s.targetOutputs())).toList();
            float aggregatedLoss = objective.calculateAggregatedLoss(batch);
            improvement = (previousLoss - aggregatedLoss) / previousLoss * -100;
            System.out.printf("[Epoch %4d] Aggregated loss = %.4f (%.2f%%)\n", iteration, aggregatedLoss, improvement);
            previousLoss = aggregatedLoss;
        }
    }

    public Float getImprovement() {
        return improvement;
    }

    public Float getPreviousLoss() {
        return previousLoss;
    }
}
