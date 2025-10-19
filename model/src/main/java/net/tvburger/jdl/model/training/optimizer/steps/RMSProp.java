package net.tvburger.jdl.model.training.optimizer.steps;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.SimpleHolder;
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

public class RMSProp<N extends Number> implements UpdateStep<LinearCombination<N>, N>, LearningRateConfigurable<N>, HyperparameterConfigurable {

    public static final String HP_BETA = "beta";

    private final Map<LinearCombination<N>, SimpleHolder<TypedVector<N>>> adaptions = new WeakHashMap<>();

    private N learningRate;
    private N beta;

    public RMSProp(N learningRate, N beta) {
        this.learningRate = learningRate;
        this.beta = beta;
    }

    @Override
    public Vector<N> calculateUpdate(Vector<N> gradients, LinearCombination<N> model, int step, Set<ExplicitRegularization<N>> regularizations) {
        JavaNumberTypeSupport<N> typeSupport = model.getCurrentNumberType();

        SimpleHolder<TypedVector<N>> accumulatedSquaredGradients = this.adaptions.computeIfAbsent(model, k -> SimpleHolder.create());
        TypedVector<N> g2 = accumulatedSquaredGradients.get();

        N[] parameters = model.getParameters();
        Vector<N> thetas = Vectors.of(model.getCurrentNumberType(), parameters).transpose();
        TypedVector<N> regularizationGradients = (TypedVector<N>) Regularizations.applyExplicitRegularization(regularizations, thetas, gradients);

        // Accumulate squared gradients
        TypedVector<N> gradientsSquared = Vectors.squared(regularizationGradients);
        TypedVector<N> secondTerm = gradientsSquared.multiply(typeSupport.min(typeSupport.one(), beta));
        g2 = g2 == null ? secondTerm : g2.multiply(beta).add(gradientsSquared);
        accumulatedSquaredGradients.set(g2);

        // Compute update
        TypedVector<N> g2Sqrt = Vectors.squareRoot(g2); // element wise sqrt
        TypedVector<N> denom = g2Sqrt.add(typeSupport.epsilon()); // epsilon is scalar, broadcasts

        return Vectors.divide(regularizationGradients, denom).multiply(model.getCurrentNumberType().negate(learningRate));
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return Map.of(
                HP_LEARNING_RATE, learningRate,
                HP_BETA, beta);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void setHyperparameter(String name, Object value) {
        if (HP_LEARNING_RATE.equals(name)) {
            this.learningRate = (N) value;
        }
        if (HP_BETA.equals(name)) {
            this.beta = (N) value;
        }
    }

    public N getBeta() {
        return beta;
    }

    public void setBeta(N beta) {
        this.beta = beta;
    }
}
