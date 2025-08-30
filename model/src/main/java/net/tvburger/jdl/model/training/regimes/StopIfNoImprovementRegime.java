package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;

public class StopIfNoImprovementRegime extends DelegatedRegime {

    private int maxEpochs;
    private int currentStalled = 0;

    public StopIfNoImprovementRegime(int maxEpochs, ObjectiveReportingRegime regime) {
        super(regime);
        this.maxEpochs = maxEpochs;
    }

    public int getMaxEpochs() {
        return maxEpochs;
    }

    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        Float improvement = ((ObjectiveReportingRegime) regime).getImprovement();
        if (improvement != null && Floats.EPSILON < improvement) {
            currentStalled++;
        } else {
            currentStalled = 0;
        }
        if (currentStalled > maxEpochs) {
            return;
        }
        regime.train(estimationFunction, trainingSet, objective, optimizer);
    }
}
