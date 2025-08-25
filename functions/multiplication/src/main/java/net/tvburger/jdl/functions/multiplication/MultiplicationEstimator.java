package net.tvburger.jdl.functions.multiplication;

import net.tvburger.dlp.DataSet;
import net.tvburger.dlp.learning.GradientDescent;
import net.tvburger.dlp.learning.loss.Losses;
import net.tvburger.dlp.nn.MultiLayerPerceptron;
import net.tvburger.dlp.nn.activations.Activations;
import net.tvburger.dlp.nn.initializers.Initializers;
import net.tvburger.dlp.utils.Floats;

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

        for (int i = 1; i < 500; i++) {
            for (int j = 1; j < 500; j++) {
                samples.add(Floats.s(Floats.a(i / 10.0f, j), Floats.a(i * j / 10.0f)));
            }
        }

        GradientDescent gradientDescent = new GradientDescent(Losses.halfMSE());
        gradientDescent.setLearningRate(0.01f);

        for (int i = 0; i < 100; i++) {
            gradientDescent.train(mlp, trainingSet);
//            mlp.dumpNodeOutputs();

            inputs = Floats.a(20.0f, 5.0f);
            estimate = mlp.estimate(inputs);
            System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(estimate));
        }
        mlp.dumpNodeOutputs();
    }

}
