package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.nn.NeuralNetwork;

import java.util.Arrays;

public class LogicalFunctions {

    public static void main(String[] args) {
        NeuralNetwork perceptron = Perceptron.create(2, 4, 1);
        perceptron.init(new PerceptronInitializer());

        DataSet dataSet = DataSets.loadOr();
        DataSet trainingSet = dataSet.subset(1, dataSet.samples().size());
        DataSet validationSet = trainingSet;
        PerceptronTrainer trainer = new PerceptronTrainer();
        for (int i = 0; i < 1000; i++) {
            trainer.train(perceptron, trainingSet);
        }
        perceptron.dumpNodeOutputs();

        for (DataSet.Sample sample : validationSet.samples()) {
            float[] estimate = perceptron.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }

    }

}
