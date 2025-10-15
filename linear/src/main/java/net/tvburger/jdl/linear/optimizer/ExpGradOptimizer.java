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
 * Custom optimizer that implements Cumulative with Reset Gradient Descent algorithm plus multiplier.
 * We keep accumulating the gradients, but reset them once we are going to oscillate.
 * In addition, there is a gamma that multiplies the accumulated gradient to increase its effect when in the same direction.
 *
 * @param <N>
 */
public class ExpGradOptimizer<N extends Number> implements LinearModelOptimizer<N>, GradientDescentOptimizer<N> {

    private boolean debugOutput;

    private final JavaNumberTypeSupport<N> typeSupport;
    private EpochCompletionListener<N> epochCompletionListener;
    private int epochs = 1000;
    private float learningRate = 0.1f;
    private float gamma = 1.1f;
    private Interceptor<N> interceptor = GradientDescentOptimizer.nullInterceptor();

    public ExpGradOptimizer(JavaNumberTypeSupport<N> typeSupport) {
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
        N[] multiplier = typeSupport.createArray(regression.getParameterCount());
        N gamma = typeSupport.valueOf(this.gamma);
        Arrays.fill(multiplier, typeSupport.one());

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
            N[] newGradients = typeSupport.createArray(adjustment.getDimensions());
            for (int i = 0; i < adjustment.getDimensions(); i++) {
                N newGradient = adjustment.get(i + 1);
                if (typeSupport.sameSign(newGradient, previousGradients[i])) {
                    multiplier[i] = typeSupport.multiply(multiplier[i], gamma);
                } else {
                    multiplier[i] = typeSupport.one();
                }
                previousGradients[i] = newGradient;
                newGradients[i] = typeSupport.multiply(newGradient, multiplier[i]);
            }
            Vector<N> adjustmentsExploded = Vectors.of(typeSupport, newGradients).transpose();
            thetas = thetas.substract(adjustmentsExploded);
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
