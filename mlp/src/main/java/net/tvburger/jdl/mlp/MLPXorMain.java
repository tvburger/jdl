package net.tvburger.jdl.mlp;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.LogicalDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.nn.training.initializers.NeuralNetworkInitializer;
import net.tvburger.jdl.model.nn.training.initializers.XavierInitializer;
import net.tvburger.jdl.model.nn.training.optimizers.NeuralNetworkOptimizers;
import net.tvburger.jdl.model.scalars.activations.Activations;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.optimizer.GradientDescentOptimizer;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.EpochRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.plots.listeners.EpochRmePlotter;

import java.util.Arrays;

public class MLPXorMain {

    public static void main(String[] args) {
        DataSet<Float> dataSet = LogicalDataSets.xor().load();
        DataSet<Float> trainingSet = dataSet;
        DataSet<Float> testSet = dataSet;

        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), Activations.sigmoid(), 2, 2, 1);

        NeuralNetworkInitializer initializer = new XavierInitializer();
        ObjectiveFunction<Float> objective = Objectives.bCE(JavaNumberTypeSupport.FLOAT);
        GradientDescentOptimizer<NeuralNetwork, Float> optimizer = NeuralNetworkOptimizers.adam();
        EpochRmePlotter epochRmePlotter = new EpochRmePlotter();
        epochRmePlotter.display();
        ChainedRegime regime = Regimes.epochs(10_000,
                        EpochRegime.sample(250, epochRmePlotter.attach("Training Set (" + trainingSet.size() + ")")),
                        EpochRegime.sample(250, epochRmePlotter.attach("Test Set (" + testSet.size() + ")", testSet)))
                .batch();
        Trainer<MultiLayerPerceptron, Float> mlpTrainer = Trainer.of(initializer, objective, optimizer, regime);
        mlpTrainer.train(mlp, trainingSet);

        NeuralNetworks.dump(mlp);

        for (DataSet.Sample<Float> sample : testSet) {
            Float[] estimate = mlp.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " vs estimated = " + Arrays.toString(estimate) + " | features " + Arrays.toString(sample.features()));
        }
    }
}
