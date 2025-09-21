package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.datasets.LinesAndCircles;
import net.tvburger.jdl.datasets.LogicalDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class AdalineMain {

    public static void main(String[] args) {
        DataSet dataSet = LogicalDataSets.toMinusSet(new LinesAndCircles().load());
        DataSet trainingSet = dataSet.subset(10, dataSet.size());
        DataSet testSet = dataSet.subset(0, 10);

        Adaline adaline = Adaline.create(400, 8);
        ObjectiveFunction objective = Objectives.mSE();
        LeastMeanSquares leastMeanSquares = new LeastMeanSquares(0.01f);
        ChainedRegime regime = Regimes.epochs(100).reportObjective().online();
        Trainer<Adaline> adalineTrainer = Trainer.of(new AdalineInitializer(), objective, leastMeanSquares, regime);
        adalineTrainer.train(adaline, trainingSet);

        for (DataSet.Sample sample : testSet) {
            boolean[] estimate = adaline.classify(sample.features());
            boolean[] booleans = Floats.toBooleans(sample.targetOutputs());
            if (Arrays.equals(booleans, estimate)) {
                System.out.println("Match! " + Arrays.toString(estimate));
            } else {
                System.out.println("MISS: real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
            }
        }
    }

}
