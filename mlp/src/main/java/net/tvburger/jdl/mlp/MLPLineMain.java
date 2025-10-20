package net.tvburger.jdl.mlp;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
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
import net.tvburger.jdl.model.training.regularization.RegularizationFactory;
import net.tvburger.jdl.model.training.regularization.Regularizations;
import net.tvburger.jdl.plots.listeners.EpochRmePlotter;

import java.util.Arrays;
import java.util.Random;

public class MLPLineMain {

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet<Float> line = SyntheticDataSets.line(-10.0f, 20.0f, JavaNumberTypeSupport.FLOAT);
        line.setNoiseScale(1.0f);
        DataSet<Float> dataSet = line.load();
        DataSet<Float> trainingSet = dataSet.resample(20, new Random());
        DataSet<Float> testSet = dataSet.resample(20, new Random());
//        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.linear(), Activations.linear(), 1, 2, 1);
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.linear(), Activations.linear(), 1, 1);

        NeuralNetworkInitializer initializer = new XavierInitializer();
        ObjectiveFunction<Float> objective = Objectives.mSE(JavaNumberTypeSupport.FLOAT);
        RegularizationFactory<Float> factory = Regularizations.getFactory(JavaNumberTypeSupport.FLOAT);
//        objective.addRegularization(factory.createElasticNet(0.5f, 0.001f));
        GradientDescentOptimizer<NeuralNetwork, Float> optimizer = NeuralNetworkOptimizers.vanilla(0.001f);
        EpochRmePlotter epochRmePlotter = new EpochRmePlotter();
        epochRmePlotter.display();
        ChainedRegime regime = Regimes.epochs(100_000,
                        EpochRegime.sample(250, epochRmePlotter.attach("Training Set (" + trainingSet.size() + ")")),
                        EpochRegime.sample(250, epochRmePlotter.attach("Test Set (" + testSet.size() + ")", testSet)))
                .batch();
        Trainer<MultiLayerPerceptron, Float> mlpTrainer = Trainer.of(initializer, objective, optimizer, regime);
        mlpTrainer.train(mlp, trainingSet);

        NeuralNetworks.dump(mlp);

        for (DataSet.Sample<Float> sample : testSet) {
            Float[] estimate = mlp.estimate(sample.features());
            System.out.println("with noise = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate) + " real = " + Arrays.toString(line.getEstimationFunction().estimate(sample.features())));
        }
    }
}
