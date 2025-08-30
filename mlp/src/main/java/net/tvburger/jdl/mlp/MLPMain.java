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
import net.tvburger.jdl.model.training.regimes.*;

import java.util.Arrays;

public class MLPMain {

    public static void main(String[] args) {
        DataSet dataSet = LogicalDataSets.xor().load();
        DataSet trainingSet = dataSet;
        DataSet testSet = dataSet;

        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), Activations.sigmoid(), 2, 2, 1);

        NeuralNetworkInitializer initializer = new XavierInitializer();
        ObjectiveFunction objective = Losses.mSE();
        StochasticGradientDescent<MultiLayerPerceptron> gradientDescent = new StochasticGradientDescent<>();
        AdamOptimizer<MultiLayerPerceptron> adamOptimizer = new AdamOptimizer<>();
        gradientDescent.setLearningRate(0.01f);
        EpochRegime epochRegime = new StopIfNoImprovementRegime(3, new ObjectiveReportingRegime(new DumpNodesRegime(Regimes.batch()))).epoch(10_000);
        Trainer<MultiLayerPerceptron> mlpTrainer = Trainer.of(initializer, objective, adamOptimizer, epochRegime);
        mlpTrainer.train(mlp, trainingSet);

        NeuralNetworks.dump(mlp);

        for (DataSet.Sample sample : testSet) {
            float[] estimate = mlp.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " vs estimated = " + Arrays.toString(estimate) + " | features " + Arrays.toString(sample.features()));
        }
    }
}
