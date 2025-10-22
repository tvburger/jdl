package net.tvburger.jdl.model.training.optimizer.steps;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.training.optimizer.LearningRateConfigurable;
import net.tvburger.jdl.model.training.optimizer.UpdateStep;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;
import net.tvburger.jdl.model.training.regularization.Regularizations;

import java.util.Map;
import java.util.Set;

public class VanillaGradientDescent<N extends Number> implements UpdateStep<LinearCombination<N>, N>, LearningRateConfigurable<N> {

    private N learningRate;

    public VanillaGradientDescent(N learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public Vector<N> calculateUpdate(Vector<N> gradients, LinearCombination<N> model, int step, Set<ExplicitRegularization<N>> regularizations) {
        Array<N> parameters = model.getParameters();
        Vector<N> thetas = new TypedVector<>(parameters, true, model.getNumberTypeSupport());
        Vector<N> regularizationGradients = Regularizations.applyExplicitRegularization(regularizations, thetas, gradients);
        return regularizationGradients.multiply(model.getNumberTypeSupport().negate(learningRate));
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
