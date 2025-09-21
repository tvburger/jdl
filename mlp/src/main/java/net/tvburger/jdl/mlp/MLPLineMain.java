package net.tvburger.jdl.mlp;

import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.nn.training.initializers.NeuralNetworkInitializer;
import net.tvburger.jdl.model.nn.training.initializers.XavierInitializer;
import net.tvburger.jdl.model.nn.training.optimizers.StochasticGradientDescent;
import net.tvburger.jdl.model.scalars.activations.Activations;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class MLPLineMain {

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet line = SyntheticDataSets.line(0.0f, 1.0f);
        line.setNoiseScale(0.0f);
        DataSet dataSet = line.load();
        DataSet trainingSet = dataSet.subset(10, dataSet.size());
        DataSet testSet = dataSet.subset(0, 10);

        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.linear(), Activations.linear(), 1, 1);

        NeuralNetworkInitializer initializer = new XavierInitializer();
        ObjectiveFunction objective = Objectives.mSE();
        StochasticGradientDescent<MultiLayerPerceptron> gradientDescent = new StochasticGradientDescent<>();
        gradientDescent.setLearningRate(0.1f);
        ChainedRegime regime = Regimes.dumpNodes().epochs(5).reportObjective().batch();
        Trainer<MultiLayerPerceptron> mlpTrainer = Trainer.of(initializer, objective, gradientDescent, regime);
        mlpTrainer.train(mlp, trainingSet);

        NeuralNetworks.dump(mlp);

        for (DataSet.Sample sample : testSet) {
            float[] estimate = mlp.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " vs estimated = " + Arrays.toString(estimate) + " | features " + Arrays.toString(sample.features()));
        }
    }
}
