package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.Matrix;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.linear.FeatureMatrices;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

import java.util.Arrays;

/**
 * Custom optimizer that implements Same Grad algorithm.
 * When the gradient of N consecutive epochs are the same, we are scaling it by N * a fraction of gamma to increase the step size.
 * We keep track of the cumulative fraction and the previous gradient.
 * When sure, the fraction grows, when less it shrinks. When the gradient is in opposite direction, it resets to 1.
 * The increase is based on a similarity score of the consecutive gradients.
 * If same we add gamma, if max diff, we subtract the cumulative fraction minus one.
 * We use a linear scale between these two extremes.
 *
 * @param <N>
 */
public class SameGradOptimizer<N extends Number> implements LinearModelOptimizer<N>, GradientDescentOptimizer<N> {

    private boolean debugOutput;

    private final JavaNumberTypeSupport<N> typeSupport;
    private EpochCompletionListener<N> epochCompletionListener;
    private int epochs = 1000;
    private float learningRate = 0.1f;
    private float gamma = 1.1f;
    private Interceptor<N> interceptor = GradientDescentOptimizer.nullInterceptor();

    public SameGradOptimizer(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    @Override
    public int getEpochs() {
        return epochs;
    }

    @Override
    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    @Override
    public float getLearningRate() {
        return learningRate;
    }

    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public EpochCompletionListener<N> getEpochCompletionListener() {
        return epochCompletionListener;
    }

    @Override
    public void setEpochCompletionListener(EpochCompletionListener<N> epochCompletionListener) {
        this.epochCompletionListener = epochCompletionListener;
    }

    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return typeSupport;
    }

    public boolean isDebugOutput() {
        return debugOutput;
    }

    public void setDebugOutput(boolean debugOutput) {
        this.debugOutput = debugOutput;
    }

    public void setGamma(float gamma) {
        this.gamma = gamma;
    }

    public float getGamma() {
        return gamma;
    }

    @Override
    public void setOptimalWeights(LinearBasisFunctionModel<N> regression, DataSet<N> trainSet) {
        JavaNumberTypeSupport<N> typeSupport = regression.getCurrentNumberType();
        if (debugOutput) {
            System.out.println("Number type = " + typeSupport.name());
        }

        N[] values = typeSupport.createArray(trainSet.size());
        for (int i = 0; i < values.length; i++) {
            values[i] = trainSet.samples().get(i).targetOutputs()[0];
        }
        TypedVector<N> y = Vectors.of(typeSupport, values).transpose();
        if (debugOutput) {
            y.print("y");
        }
        Matrix<N> designMatrix = FeatureMatrices.create(regression.getFeatureExtractor(), trainSet);
        if (debugOutput) {
            designMatrix.print("Φ");
        }

        N learningRate = typeSupport.valueOf(this.learningRate);
        N trainingSetSizeInverse = typeSupport.divide(typeSupport.one(), typeSupport.valueOf(trainSet.size()));
        N[] previousGradients = typeSupport.createArray(regression.getParameterCount());
        N[] accumulativeGammas = typeSupport.createArray(regression.getParameterCount());
        N[] newGradients = typeSupport.createArray(regression.getParameterCount());
        Arrays.fill(accumulativeGammas, typeSupport.one());
        N gamma = typeSupport.valueOf(this.gamma);

        for (int e = 1; e <= epochs; e++) {
            N[] parameters = regression.getParameters();
            Vector<N> thetas = Vectors.of(regression.getCurrentNumberType(), parameters).transpose();
            if (debugOutput) {
                thetas.print("θ");
            }
            Vector<N> error = designMatrix.multiply(thetas).substract(y);
            if (debugOutput) {
                error.print("Error");
            }
            Vector<N> adjustment = interceptor.interceptGradients(e, regression, trainSet, this, designMatrix.transpose().multiply(error).multiply(trainingSetSizeInverse)).multiply(learningRate);
            if (debugOutput) {
                adjustment.print("Adjustment");
            }
            for (int i = 0; i < previousGradients.length; i++) {
                N newGradient = adjustment.get(i + 1);
                if (!typeSupport.isZero(previousGradients[i]) && typeSupport.sameSign(newGradient, previousGradients[i])) {
                    N linearSameScore = typeSupport.subtract(
                            typeSupport.one(),
                            typeSupport.divide(
                                    typeSupport.min(
                                            typeSupport.absolute(typeSupport.subtract(previousGradients[i], newGradient)),
                                            previousGradients[i]),
                                    previousGradients[i]));
                    accumulativeGammas[i] = typeSupport.add(
                            typeSupport.multiply(
                                    linearSameScore,
                                    typeSupport.add(
                                            typeSupport.subtract(
                                                    accumulativeGammas[i],
                                                    typeSupport.one()),
                                            gamma)),
                            typeSupport.one());
                    newGradients[i] = typeSupport.multiply(newGradient, accumulativeGammas[i]);
                } else {
                    accumulativeGammas[i] = typeSupport.one();
                    newGradients[i] = newGradient;
                }
                previousGradients[i] = newGradient;
            }
            Vector<N> adjustmentWithAccumulatedGradients = Vectors.of(typeSupport, newGradients).transpose();
            thetas = thetas.substract(adjustmentWithAccumulatedGradients);
            regression.setParameters(thetas.asArray());
            if (epochCompletionListener != null) {
                epochCompletionListener.epochCompleted(e, regression, trainSet, this);
            }
        }
    }

    @Override
    public void setInterceptor(Interceptor<N> interceptor) {
        this.interceptor = interceptor == null ? GradientDescentOptimizer.nullInterceptor() : interceptor;
    }

    @Override
    public Interceptor<N> getInterceptor() {
        return interceptor;
    }
}
