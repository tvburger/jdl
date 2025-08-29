package net.tvburger.jdl.mlp;

import net.tvburger.jdl.datasets.LogicalDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.learning.BatchTrainer;
import net.tvburger.jdl.model.learning.EpochTrainer;
import net.tvburger.jdl.model.loss.Losses;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.nn.activations.Activations;
import net.tvburger.jdl.model.nn.initializers.XavierInitializer;
import net.tvburger.jdl.model.nn.optimizer.GradientDescent;

import java.util.Arrays;

public class MLPMain {

    public static void main(String[] args) {
        DataSet dataSet = LogicalDataSets.xor().load();
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), Activations.sigmoid(), 2, 2, 1);
        mlp.init(new XavierInitializer());

        DataSet trainingSet = dataSet;
        DataSet testSet = dataSet;

        GradientDescent<MultiLayerPerceptron> gradientDescent = new GradientDescent<>(Losses.bCE());
        gradientDescent.setLearningRate(0.1f);

        EpochTrainer<MultiLayerPerceptron> trainer = new EpochTrainer<>(new BatchTrainer<>(gradientDescent));
        trainer.setEpochs(1000000);
        NeuralNetworks.dump(mlp);
        trainer.train(mlp, trainingSet);
        NeuralNetworks.dump(mlp);

        for (DataSet.Sample sample : testSet) {
            float[] estimate = mlp.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " vs estimated = " + Arrays.toString(estimate) + " | features " + Arrays.toString(sample.features()));
        }
    }
}
