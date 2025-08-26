package net.tvburger.jdl.functions.multiplication;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.learning.GradientDescent;
import net.tvburger.jdl.learning.loss.Losses;
import net.tvburger.jdl.nn.activations.Activations;
import net.tvburger.jdl.nn.initializers.Initializers;
import net.tvburger.jdl.utils.Floats;
import net.tvburger.jdl.mlp.MultiLayerPerceptron;

import java.util.ArrayList;
import java.util.Arrays;

public class MultiplicationEstimator {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.linear(), 2, 10, 1);
        System.out.println(Arrays.toString(mlp.getParameters()));
        mlp.init(Initializers.random());
        System.out.println(Arrays.toString(mlp.getParameters()));

        float[] inputs = new float[]{20.0f, 5.0f};
        mlp.dumpNodeOutputs();
        float[] estimate = mlp.estimate(inputs);
        System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(estimate));

        ArrayList<DataSet.Sample> samples = new ArrayList<>();
        DataSet trainingSet = new DataSet(samples);

        for (int i = 1; i < 50; i++) {
            for (int j = 1; j < 50; j++) {
                samples.add(Floats.s(Floats.a(i / 10.0f, j), Floats.a(i * j / 10.0f)));
            }
        }

        GradientDescent gradientDescent = new GradientDescent(Losses.halfMSE());
        gradientDescent.setLearningRate(0.0001f);

        for (int i = 0; i < 100; i++) {
            for (DataSet.Sample sample : trainingSet.samples()) {
                gradientDescent.train(mlp, DataSet.of(sample));

            }
            inputs = Floats.a(3.5f, 2.8f);
            estimate = mlp.estimate(inputs);
            System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(estimate));
        }
        mlp.dumpNodeOutputs();
    }

}
