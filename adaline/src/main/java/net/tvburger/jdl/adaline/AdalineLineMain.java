package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class AdalineLineMain {

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet<Float> syntheticDataSet = SyntheticDataSets.line(7, 3, JavaNumberTypeSupport.FLOAT);
        DataSet<Float> dataSet = syntheticDataSet.load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);

        Adaline adaline = Adaline.create(1, 1);
        ObjectiveFunction objective = Objectives.mSE();
        LeastMeanSquares leastMeanSquares = new LeastMeanSquares(0.001f);
        ChainedRegime regime = Regimes.epochs(5).dumpNodes().reportObjective().online();
        Trainer<Adaline> adalineTrainer = Trainer.of(new AdalineInitializer(), objective, leastMeanSquares, regime);
        adalineTrainer.train(adaline, trainingSet);
        NeuralNetworks.dump(adaline);

        for (DataSet.Sample<Float> sample : testSet) {
            Float[] estimate = adaline.estimate(sample.features());
            System.out.println("with noise = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate) + " real = " + Arrays.toString(syntheticDataSet.getEstimationFunction().estimate(sample.features())));
        }
    }

}
