package net.tvburger.jdl.mlp;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.nn.training.initializers.NeuralNetworkInitializer;
import net.tvburger.jdl.model.nn.training.initializers.XavierInitializer;
import net.tvburger.jdl.model.nn.training.optimizers.BackPropagation;
import net.tvburger.jdl.model.nn.training.optimizers.NeuralNetworkOptimizers;
import net.tvburger.jdl.model.scalars.activations.Activations;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.optimizer.GradientDescentOptimizer;
import net.tvburger.jdl.model.training.optimizer.steps.AdamW;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class MLPLineMain {

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet<Float> line = SyntheticDataSets.line(0.0f, 1.0f, JavaNumberTypeSupport.FLOAT);
        line.setNoiseScale(0.1f);
        DataSet<Float> dataSet = line.load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);

        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.linear(), Activations.linear(), 1, 1);

        NeuralNetworkInitializer initializer = new XavierInitializer();
        ObjectiveFunction<Float> objective = Objectives.mSE(JavaNumberTypeSupport.FLOAT);
        GradientDescentOptimizer<NeuralNetwork, Float> optimizer = NeuralNetworkOptimizers.adamW();
        ChainedRegime regime = Regimes.dumpNodes().epochs(1_000).reportObjective().batch();
        Trainer<MultiLayerPerceptron, Float> mlpTrainer = Trainer.of(initializer, objective, optimizer, regime);
        mlpTrainer.train(mlp, trainingSet);

        NeuralNetworks.dump(mlp);

        for (DataSet.Sample<Float> sample : testSet) {
            Float[] estimate = mlp.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " vs estimated = " + Arrays.toString(estimate) + " | features " + Arrays.toString(sample.features()));
        }
    }
}
