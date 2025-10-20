package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.HyperparameterConfigurable;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.TrainableFunction;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class GradientDescentOptimizer<E extends TrainableFunction<N>, N extends Number> implements Optimizer<E, N>, HyperparameterConfigurable {

    public boolean debug = false;

    private final ObjectiveGradientEstimator<N> objectiveGradientEstimator;
    private final GradientDescentModelDecomposer<E, N> modelDecomposer;
    private final UpdateStep<LinearCombination<N>, N> updateStep;

    public GradientDescentOptimizer(GradientDescentModelDecomposer<E, N> modelDecomposer, UpdateStep<LinearCombination<N>, N> updateStep) {
        this(new ObjectiveGradientEstimator<>(), modelDecomposer, updateStep);
    }

    protected GradientDescentOptimizer(ObjectiveGradientEstimator<N> objectiveGradientEstimator, GradientDescentModelDecomposer<E, N> modelDecomposer, UpdateStep<LinearCombination<N>, N> updateStep) {
        this.objectiveGradientEstimator = objectiveGradientEstimator;
        this.modelDecomposer = modelDecomposer;
        this.updateStep = updateStep;
    }

    @Override
    public void optimize(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, int step) {
        Map<LinearCombination<N>, Vector<N>> accumulatedAdjustments = new HashMap<>();
        N trainingSetSize = estimationFunction.getCurrentNumberType().valueOf(trainingSet.size());
        for (DataSet.Sample<N> sample : trainingSet) {
            Vector<N> objectiveGradients = objectiveGradientEstimator.determineGradient(sample, estimationFunction, objective);
            modelDecomposer.calculateDecompositionGradients(estimationFunction, objectiveGradients, sample.features())
                    .forEach(d -> accumulatedAdjustments.merge(d.linearCombination(), d.parameterGradients(), Vector::add));
        }

        accumulatedAdjustments.forEach((m, a) -> {
            if (debug && step == 1) {
                System.out.println("0: Applying accumulated adjustment for model: " + Arrays.toString(m.getParameters()));
            }
            N[] parameters = m.getParameters();
            Vector<N> meanGradients = a.divide(trainingSetSize);
            Vector<N> adjustments = updateStep.calculateUpdate(meanGradients, m, step, objective.getRegularizations());
            Vector<N> thetas = Vectors.of(m.getCurrentNumberType(), parameters).transpose();
            Vector<N> updatedThetas = thetas.add(adjustments);
            N[] updatedParameters = updatedThetas.asArray();
            m.setParameters(updatedParameters);
            if (debug) {
                System.out.println(step + ": Applied accumulated adjustment for model: " + Arrays.toString(m.getParameters()));
            }
        });
    }

    public UpdateStep<LinearCombination<N>, N> getUpdateStep() {
        return updateStep;
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        if (updateStep instanceof HyperparameterConfigurable configurable) {
            return configurable.getHyperparameters();
        }
        return Map.of();
    }

    @Override
    public void setHyperparameter(String name, Object value) {
        if (updateStep instanceof HyperparameterConfigurable configurable) {
            configurable.setHyperparameter(name, value);
        }
    }
}
