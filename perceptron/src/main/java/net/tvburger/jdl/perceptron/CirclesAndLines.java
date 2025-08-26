package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.nn.NeuralNetwork;

import java.util.Arrays;

public class CirclesAndLines {

    public static void main(String[] args) {
        NeuralNetwork perceptron = Perceptron.create();
        perceptron.init(new PerceptronInitializer());

        DataSet dataSet = DataSets.loadLinesAndCircles();
        DataSet trainingSet = dataSet.subset(11, dataSet.samples().size());
        DataSet validationSet = dataSet.subset(1, 11);
        PerceptronTrainer trainer = new PerceptronTrainer();
        for (int i = 0; i < 10; i++) {
            trainer.train(perceptron, trainingSet);
        }
        perceptron.dumpNodeOutputs();

        for (DataSet.Sample sample : validationSet.samples()) {
            float[] estimate = perceptron.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }

    }

}
