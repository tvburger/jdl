package net.tvburger.jdl.functions.xor;

import net.tvburger.jdl.mlp.MultiLayerPerceptron;
import net.tvburger.jdl.nn.activations.Activations;
import net.tvburger.jdl.nn.initializers.Initializers;

import java.util.Arrays;

public class XorEstimator {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), 2, 2, 1);
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
