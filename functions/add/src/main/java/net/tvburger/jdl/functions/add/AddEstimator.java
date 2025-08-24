package net.tvburger.jdl.functions.add;

import net.tvburger.dlp.initializers.Initializers;
import net.tvburger.dlp.nn.MultiLayerPerceptron;

import java.util.Arrays;

public class AddEstimator {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(2, 1);
        System.out.println(Arrays.toString(mlp.getParameters()));
        mlp.init(Initializers.random());
        System.out.println(Arrays.toString(mlp.getParameters()));

        float[] inputs = new float[]{1.0f, 0.0f};
        mlp.dumpNodeOutputs();
        float[] estimate = mlp.estimate(inputs);
        mlp.dumpNodeOutputs();

        System.out.println(Arrays.toString(inputs) + " -> " + Arrays.toString(estimate));
    }

}
