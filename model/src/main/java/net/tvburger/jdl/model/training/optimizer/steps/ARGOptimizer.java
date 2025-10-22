package net.tvburger.jdl.model.training.optimizer.steps;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.training.optimizer.LearningRateConfigurable;
import net.tvburger.jdl.model.training.optimizer.UpdateStep;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;
import net.tvburger.jdl.model.training.regularization.Regularizations;

import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

/**
 * Custom optimizer that implements Cumulative with Reset Gradient Descent algorithm.
 * We keep accumulating the parameterGradients, but reset them once we are going to oscillate.
 *
 * @param <N>
 */
public class ARGOptimizer<N extends Number> implements UpdateStep<LinearCombination<N>, N>, LearningRateConfigurable<N> {

    private final Map<LinearCombination<N>, Array<N>> accumulatedGradients = new WeakHashMap<>();

    private N learningRate;

    public ARGOptimizer(N learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public Vector<N> calculateUpdate(Vector<N> gradients, LinearCombination<N> model, int step, Set<ExplicitRegularization<N>> regularizations) {
        Array<N> accumulatedGradients = this.accumulatedGradients.computeIfAbsent(model, k -> model.getNumberTypeSupport().createArray(model.getParameterCount()));
        JavaNumberTypeSupport<N> typeSupport = model.getNumberTypeSupport();

        Array<N> parameters = model.getParameters();
        Vector<N> thetas = new TypedVector<>(parameters, true, model.getNumberTypeSupport());
        Vector<N> regularizationGradients = Regularizations.applyExplicitRegularization(regularizations, thetas, gradients);

        for (int i = 0; i < regularizationGradients.getDimensions(); i++) {
            N newGradient = regularizationGradients.get(i + 1);
            if (typeSupport.hasSameSign(newGradient, accumulatedGradients.get(i))) {
                accumulatedGradients.set(i, typeSupport.add(accumulatedGradients.get(i), newGradient));
            } else {
                accumulatedGradients.set(i, newGradient);
            }
        }

        return new TypedVector<>(accumulatedGradients, true, model.getNumberTypeSupport()).multiply(model.getNumberTypeSupport().negate(learningRate));
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return Map.of(HP_LEARNING_RATE, learningRate);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void setHyperparameter(String name, Object value) {
        if (HP_LEARNING_RATE.equals(name)) {
            this.learningRate = (N) value;
        }
    }
}
