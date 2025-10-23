package net.tvburger.jdl.model.nn.training.optimizers;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.StaticUtility;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.training.optimizer.GradientDescentOptimizer;
import net.tvburger.jdl.model.training.optimizer.steps.*;

@StaticUtility
public final class NeuralNetworkOptimizers {

    private static final BackPropagation BACK_PROPAGATION = new BackPropagation();

    public static final float DEFAULT_LEARNING_RATE = 0.1f;
    public static final float DEFAULT_ADAPTIVE_BETA = 0.9f;
    public static final float DEFAULT_MOMENTUM_BETA = 0.999f;
    public static final float DEFAULT_LAMBDA = 0.001f;

    private NeuralNetworkOptimizers() {
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> vanilla() {
        return vanilla(DEFAULT_LEARNING_RATE);
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> vanilla(float learningRate) {
        return new GradientDescentOptimizer<>(BACK_PROPAGATION, new VanillaGradientDescent<>(JavaNumberTypeSupport.FLOAT, learningRate));
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> adaGrad() {
        return adaGrad(DEFAULT_LEARNING_RATE);
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> adaGrad(float learningRate) {
        return new GradientDescentOptimizer<>(BACK_PROPAGATION, new AdaGrad<>(learningRate));
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> rmsProp() {
        return rmsProp(DEFAULT_LEARNING_RATE, DEFAULT_ADAPTIVE_BETA);
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> rmsProp(float learningRate, float beta) {
        return new GradientDescentOptimizer<>(BACK_PROPAGATION, new RMSProp<>(learningRate, beta));
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> adam() {
        return adam(DEFAULT_LEARNING_RATE, DEFAULT_ADAPTIVE_BETA, DEFAULT_MOMENTUM_BETA);
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> adam(float learningRate, float beta1, float beta2) {
        return new GradientDescentOptimizer<>(BACK_PROPAGATION, new Adam<>(learningRate, beta1, beta2));
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> adamW() {
        return adamW(DEFAULT_LEARNING_RATE, DEFAULT_ADAPTIVE_BETA, DEFAULT_MOMENTUM_BETA, DEFAULT_LAMBDA);
    }

    public static GradientDescentOptimizer<NeuralNetwork, Float> adamW(float learningRate, float beta1, float beta2, float lambda) {
        return new GradientDescentOptimizer<>(BACK_PROPAGATION, new AdamW<>(learningRate, beta1, beta2, lambda));
    }
}
