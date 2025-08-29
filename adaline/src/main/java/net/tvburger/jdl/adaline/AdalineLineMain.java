package net.tvburger.jdl.adaline;

import net.tvburger.jdl.datasets.StraightLineWithNoise;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Losses;
import net.tvburger.jdl.model.training.regimes.EpochRegime;
import net.tvburger.jdl.model.training.regimes.ObjectiveReportingRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class AdalineLineMain {

    public static void main(String[] args) {
        DataSet dataSet = new StraightLineWithNoise().load();
        DataSet trainingSet = dataSet.subset(11, dataSet.samples().size());
        DataSet testSet = dataSet.subset(1, 11);

        Adaline adaline = Adaline.create(1, 1);
        ObjectiveFunction objective = Losses.mSE();
        LeastMeanSquares leastMeanSquares = new LeastMeanSquares(0.00001f);
        EpochRegime epochRegime = new ObjectiveReportingRegime(Regimes.online()).epoch(200);
        Trainer<Adaline> adalineTrainer = Trainer.of(new AdalineInitializer(), objective, leastMeanSquares, epochRegime);
        adalineTrainer.train(adaline, trainingSet);
        NeuralNetworks.dump(adaline);

        for (DataSet.Sample sample : testSet) {
            float[] estimate = adaline.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }
    }

}
