package net.tvburger.jdl.mlp;

import net.tvburger.jdl.datasets.LogicalDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.nn.activations.Activations;
import net.tvburger.jdl.model.nn.initializers.NeuralNetworkInitializer;
import net.tvburger.jdl.model.nn.initializers.XavierInitializer;
import net.tvburger.jdl.model.nn.optimizers.AdamOptimizer;
import net.tvburger.jdl.model.nn.optimizers.StochasticGradientDescent;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Losses;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class MLPMain {

    public static void main(String[] args) {
        DataSet dataSet = LogicalDataSets.or().load();
        DataSet trainingSet = dataSet.subset(1, 3);
        DataSet testSet = dataSet;

        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), Activations.sigmoid(), 2, 1, 1, 1);

        NeuralNetworkInitializer initializer = new XavierInitializer();
        ObjectiveFunction objective = Losses.mSE();
        StochasticGradientDescent<MultiLayerPerceptron> gradientDescent = new StochasticGradientDescent<>();
        AdamOptimizer<MultiLayerPerceptron> adamOptimizer = new AdamOptimizer<>();
        gradientDescent.setLearningRate(0.01f);
        ChainedRegime regime = Regimes.chainTop().epochs(10_000).stopIfNoImprovements(true).dumpNodes().batch().bottomChain();
        Trainer<MultiLayerPerceptron> mlpTrainer = Trainer.of(initializer, objective, adamOptimizer, regime);
        mlpTrainer.train(mlp, trainingSet);

        NeuralNetworks.dump(mlp);

        for (DataSet.Sample sample : testSet) {
            float[] estimate = mlp.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " vs estimated = " + Arrays.toString(estimate) + " | features " + Arrays.toString(sample.features()));
        }
    }
}
