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

        DataSet<Float> dataSet = new LinesAndCircles().load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> validationSet = dataSet.subset(0, 10);
        Regime regime = Regimes.epochs(10).online();
        Trainer<Perceptron> trainer = Trainer.of(new PerceptronInitializer(), null, new PerceptronUpdateRule(), regime);
        trainer.train(mark1, trainingSet);

        for (DataSet.Sample<Float> sample : validationSet) {
            Float[] estimate = mark1.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }
    }

}
