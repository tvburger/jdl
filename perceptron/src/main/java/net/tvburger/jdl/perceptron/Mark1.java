package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.datasets.LinesAndCircles;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.regimes.Regimes;

import java.util.Arrays;

public class Mark1 {

    public static void main(String[] args) {
        Perceptron mark1 = Perceptron.create(400, 512, 8);
        mark1.accept(new PerceptronInitializer());

        DataSet dataSet = new LinesAndCircles().load();
        DataSet trainingSet = dataSet.subset(10, dataSet.size());
        DataSet validationSet = dataSet.subset(0, 10);
        Regime regime = Regimes.epochs(10).online();
        Trainer<Perceptron> trainer = Trainer.of(new PerceptronInitializer(), null, new PerceptronUpdateRule(), regime);
        trainer.train(mark1, trainingSet);

        for (DataSet.Sample sample : validationSet) {
            float[] estimate = mark1.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }
    }

}
