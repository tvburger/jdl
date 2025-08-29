package net.tvburger.jdl.adaline;

import net.tvburger.jdl.datasets.StraightLineWithNoise;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.learning.EpochTrainer;
import net.tvburger.jdl.model.nn.NeuralNetworks;

import java.util.Arrays;

public class AdalineLineMain {

    public static void main(String[] args) {
        DataSet dataSet = new StraightLineWithNoise().load();
        DataSet trainingSet = dataSet.subset(11, dataSet.samples().size());
        DataSet testSet = dataSet.subset(1, 11);

        Adaline adaline = Adaline.create(1, 1);
        adaline.init(new AdalineInitializer());

        EpochTrainer<Adaline> trainer = new EpochTrainer<>(new AdalineTrainingFunction(0.0001f).asTrainer());
        trainer.setEpochs(100);
        trainer.train(adaline, trainingSet);
        NeuralNetworks.dump(adaline);

        for (DataSet.Sample sample : testSet) {
            float[] estimate = adaline.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }
    }

}
