package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.datasets.LinesAndCircles;
import net.tvburger.jdl.datasets.LogicalDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Losses;
import net.tvburger.jdl.model.training.regimes.EpochRegime;
import net.tvburger.jdl.model.training.regimes.ObjectiveReportingRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class AdalineMain {

    public static void main(String[] args) {
        DataSet dataSet = LogicalDataSets.toMinusSet(new LinesAndCircles().load());
        DataSet trainingSet = dataSet.subset(11, dataSet.samples().size());
        DataSet testSet = dataSet.subset(1, 11);

        Adaline adaline = Adaline.create(400, 8);
        ObjectiveFunction objective = Losses.mSE();
        LeastMeanSquares leastMeanSquares = new LeastMeanSquares(0.01f);
        EpochRegime epochRegime = new ObjectiveReportingRegime(Regimes.online()).epoch(100);
        Trainer<Adaline> adalineTrainer = Trainer.of(new AdalineInitializer(), objective, leastMeanSquares, epochRegime);
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
