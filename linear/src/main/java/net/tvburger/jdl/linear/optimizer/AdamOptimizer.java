package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.Matrix;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.linear.FeatureMatrices;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.DataSet;

import java.lang.reflect.Type;

public class AdamOptimizer<N extends Number> implements LinearModelOptimizer<N>, GradientDescentOptimizer<N> {

    private boolean debugOutput;

    private final JavaNumberTypeSupport<N> typeSupport;
    private EpochCompletionListener<N> epochCompletionListener;
    private int epochs = 1000;
    private float learningRate = 0.1f;
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private Interceptor<N> interceptor = GradientDescentOptimizer.nullInterceptor();

    public AdamOptimizer(JavaNumberTypeSupport<N> typeSupport) {
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

    public float getBeta1() {
        return beta1;
    }

    public void setBeta1(float beta1) {
        this.beta1 = beta1;
    }

    public float getBeta2() {
        return beta2;
    }

    public void setBeta2(float beta2) {
        this.beta2 = beta2;
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

        TypedVector<N> m = Vectors.of(typeSupport, typeSupport.createArray(regression.getParameterCount()));
        TypedVector<N> v = Vectors.of(typeSupport, typeSupport.createArray(regression.getParameterCount()));
        N beta1 = typeSupport.valueOf(this.beta1);
        N beta2 = typeSupport.valueOf(this.beta2);
        N epsilon = typeSupport.valueOf(1.0e-8); // small constant to prevent division by zero

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
            Vector<N> gradients = interceptor.interceptGradients(e, regression, trainSet, this, designMatrix.transpose().multiply(error).multiply(trainingSetSizeInverse));

            // m = β1 * m + (1 - β1) * g
            m = m.multiply(beta1).add(gradients.multiply(typeSupport.subtract(typeSupport.one(), beta1)));

            // v = β2 * v + (1 - β2) * g^2
            Vector<N> gradientSquared = Vectors.squared((TypedVector<N>) gradients);
            v = v.multiply(beta2).add(gradientSquared.multiply(typeSupport.subtract(typeSupport.one(), beta2)));

            // Bias correction
            double beta1T = Math.pow(this.beta1, e);
            double beta2T = Math.pow(this.beta2, e);
            N oneMinusBeta1T = typeSupport.valueOf(1.0 - beta1T);
            N oneMinusBeta2T = typeSupport.valueOf(1.0 - beta2T);

            TypedVector<N> mHat = m.multiply(typeSupport.inverse(oneMinusBeta1T));
            TypedVector<N> vHat = v.multiply(typeSupport.inverse(oneMinusBeta2T));

            // θ = θ - α * m̂ / (sqrt(v̂) + ε)
            TypedVector<N> vHatSqrt = Vectors.squareRoot(vHat); // assumes sqrt() method; else do element-wise sqrt
            TypedVector<N> denom = vHatSqrt.add(epsilon); // epsilon is scalar, broadcasts

            Vector<N> adjustment = Vectors.divide(mHat, denom).multiply(learningRate);
            if (debugOutput) {
                adjustment.print("Adjustment");
            }
            thetas = thetas.substract(adjustment);
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
