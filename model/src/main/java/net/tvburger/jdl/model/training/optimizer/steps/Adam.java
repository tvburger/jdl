package net.tvburger.jdl.model.training.optimizer.steps;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.model.HyperparameterConfigurable;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.training.optimizer.LearningRateConfigurable;
import net.tvburger.jdl.model.training.optimizer.UpdateStep;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;
import net.tvburger.jdl.model.training.regularization.Regularizations;

import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

public class Adam<N extends Number> implements UpdateStep<LinearCombination<N>, N>, LearningRateConfigurable<N>, HyperparameterConfigurable {

    public static final String HP_BETA_1 = "beta1";
    public static final String HP_BETA_2 = "beta2";

    private final Map<LinearCombination<N>, Pair<TypedVector<N>, TypedVector<N>>> adaptions = new WeakHashMap<>();

    private N learningRate;
    private N beta1;
    private N beta2;

    public Adam(N learningRate, N beta1, N beta2) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public Vector<N> calculateUpdate(Vector<N> gradients, LinearCombination<N> model, int step, Set<ExplicitRegularization<N>> regularizations) {
        JavaNumberTypeSupport<N> typeSupport = model.getCurrentNumberType();

        Pair<TypedVector<N>, TypedVector<N>> adaptions = this.adaptions.computeIfAbsent(model, k -> Pair.mutable(null, null));
        TypedVector<N> m = adaptions.left();
        TypedVector<N> v = adaptions.right();

        N[] parameters = model.getParameters();
        Vector<N> thetas = Vectors.of(model.getCurrentNumberType(), parameters).transpose();
        TypedVector<N> regularizationGradients = (TypedVector<N>) Regularizations.applyExplicitRegularization(regularizations, thetas, gradients);

        // m = β1 * m + (1 - β1) * g
        TypedVector<N> secondTermM = regularizationGradients.multiply(typeSupport.subtract(typeSupport.one(), beta1));
        m = m == null ? secondTermM : m.multiply(beta1).add(secondTermM);

        // v = β2 * v + (1 - β2) * g^2
        TypedVector<N> gradientSquared = Vectors.squared(regularizationGradients);
        TypedVector<N> secondTermV = gradientSquared.multiply(typeSupport.subtract(typeSupport.one(), beta2));
        v = v == null ? secondTermV : v.multiply(beta2).add(secondTermV);

        // Store the results
        adaptions.setLeft(m);
        adaptions.setRight(v);

        // Bias correction
        N beta1T = typeSupport.pow(this.beta1, step);
        N beta2T = typeSupport.pow(this.beta2, step);
        N oneMinusBeta1T = typeSupport.subtract(typeSupport.one(), beta1T);
        N oneMinusBeta2T = typeSupport.subtract(typeSupport.one(), beta2T);

        TypedVector<N> mHat = m.multiply(typeSupport.inverse(oneMinusBeta1T));
        TypedVector<N> vHat = v.multiply(typeSupport.inverse(oneMinusBeta2T));

        // θ = θ - α * m̂ / (sqrt(v̂) + ε)
        TypedVector<N> vHatSqrt = Vectors.squareRoot(vHat); // element wise sqrt
        TypedVector<N> denom = vHatSqrt.add(typeSupport.epsilon()); // epsilon is scalar, broadcasts

        return Vectors.divide(mHat, denom).multiply(model.getCurrentNumberType().negate(learningRate));
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return Map.of(
                HP_LEARNING_RATE, learningRate,
                HP_BETA_1, beta1,
                HP_BETA_2, beta2);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void setHyperparameter(String name, Object value) {
        if (HP_LEARNING_RATE.equals(name)) {
            this.learningRate = (N) value;
        }
        if (HP_BETA_1.equals(name)) {
            this.beta1 = (N) value;
        }
        if (HP_BETA_2.equals(name)) {
            this.beta2 = (N) value;
        }
    }

    public N getBeta1() {
        return beta1;
    }

    public void setBeta1(N beta1) {
        this.beta1 = beta1;
    }

    public N getBeta2() {
        return beta2;
    }

    public void setBeta2(N beta2) {
        this.beta2 = beta2;
    }
}
