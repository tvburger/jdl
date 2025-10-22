package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

public class AdalineLineMain {

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet<Float> syntheticDataSet = SyntheticDataSets.line(2, 3, JavaNumberTypeSupport.FLOAT);
        DataSet<Float> dataSet = syntheticDataSet.load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);

        Adaline adaline = Adaline.create(1, 1);
        ObjectiveFunction<Float> objective = Objectives.mSE(JavaNumberTypeSupport.FLOAT);
        LeastMeanSquares leastMeanSquares = new LeastMeanSquares(0.001f);
        ChainedRegime regime = Regimes.epochs(100).dumpNodes().reportObjective().stochastic();
        Trainer<Adaline, Float> adalineTrainer = Trainer.of(new AdalineInitializer(), objective, leastMeanSquares, regime);
        adalineTrainer.train(adaline, trainingSet);
        NeuralNetworks.dump(adaline);

        for (DataSet.Sample<Float> sample : testSet) {
            Array<Float> estimate = adaline.estimate(sample.features());
            System.out.println("with noise = " + Array.toString(sample.targetOutputs()) + " estimated = " + Array.toString(estimate) + " real = " + Array.toString(syntheticDataSet.getEstimationFunction().estimate(sample.features())));
        }
    }

}
