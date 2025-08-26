package net.tvburger.jdl.functions.add;

import net.tvburger.jdl.DataSet;
import net.tvburger.jdl.learning.GradientDescent;
import net.tvburger.jdl.learning.loss.Losses;
import net.tvburger.jdl.mlp.MultiLayerPerceptron;
import net.tvburger.jdl.nn.activations.Activations;
import net.tvburger.jdl.nn.initializers.Initializers;
import net.tvburger.jdl.utils.Floats;

import java.util.Arrays;
import java.util.List;

public class AddEstimator {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.reLU(), 2, 1);
        System.out.println(Arrays.toString(mlp.getParameters()));
        mlp.init(Initializers.random());
        System.out.println(Arrays.toString(mlp.getParameters()));

        float[] inputs = new float[]{5.0f, 5.0f};
        mlp.dumpNodeOutputs();
        float[] estimate = mlp.estimate(inputs);
        System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(estimate));

        DataSet trainingSet = new DataSet(List.of(
                Floats.s(Floats.a(1.0f, 3.0f), Floats.a(4.0f)),
                Floats.s(Floats.a(2.0f, 2.0f), Floats.a(4.0f)),
                Floats.s(Floats.a(10.0f, 20.0f), Floats.a(30.0f)),
                Floats.s(Floats.a(0.5f, 0.7f), Floats.a(1.2f)),
                Floats.s(Floats.a(-6f, 20.0f), Floats.a(14.0f)),
                Floats.s(Floats.a(30.0f, 8.0f), Floats.a(38.0f))
        ));

        GradientDescent gradientDescent = new GradientDescent(Losses.halfMSE());
        gradientDescent.setLearningRate(0.0001f);

        for (int i = 0; i < 100; i++) {
            for (DataSet.Sample sample : trainingSet.samples()) {
                gradientDescent.train(mlp, DataSet.of(sample));
            }

            inputs = Floats.a(5.0f, 5.0f);
            estimate = mlp.estimate(inputs);
            System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(estimate));
        }
        mlp.dumpNodeOutputs();
    }

}
