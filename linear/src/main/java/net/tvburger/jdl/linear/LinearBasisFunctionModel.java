package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.HyperparameterConfigurable;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.scalars.TrainableScalarFunction;
import net.tvburger.jdl.model.scalars.UnaryEstimationFunction;

import java.util.HashMap;
import java.util.Map;

@Strategy(Strategy.Role.CONCRETE)
public class LinearBasisFunctionModel implements LinearModel, TrainableScalarFunction, HyperparameterConfigurable, UnaryEstimationFunction {

    public static final String HP_M = "M";

    private final Map<String, Object> hyperparameters;
    private final BasisFunction.Generator basisFunctionGenerator;
    private FeatureExtractor featureExtractor;
    private LinearCombination linearCombination;

    private static void ensureValidM(int m) {
        if (m < 0) {
            throw new IllegalArgumentException("M must be >= 0!");
        }
    }

    public static LinearBasisFunctionModel create(int m, BasisFunction.Generator basisFunctionGenerator) {
        return new LinearBasisFunctionModel(m, basisFunctionGenerator);
    }

    private static Map<String, Object> createHyperparameters(int m) {
        ensureValidM(m);
        Map<String, Object> hyperparameters = new HashMap<>();
        hyperparameters.put(HP_M, m);
        return hyperparameters;
    }

    protected LinearBasisFunctionModel(int m, BasisFunction.Generator basisFunctionGenerator) {
        this(createHyperparameters(m), basisFunctionGenerator, basisFunctionGenerator.generate(m), LinearCombination.create(m));
    }

    public LinearBasisFunctionModel(Map<String, Object> hyperparameters, BasisFunction.Generator basisFunctionGenerator, FeatureExtractor featureExtractor, LinearCombination linearCombination) {
        this.hyperparameters = hyperparameters;
        this.basisFunctionGenerator = basisFunctionGenerator;
        this.featureExtractor = featureExtractor;
        this.linearCombination = linearCombination;
    }

    public FeatureExtractor getFeatureExtractor() {
        return featureExtractor;
    }

    public int getModelComplexity() {
        return (Integer) hyperparameters.get(HP_M);
    }

    public void setModelComplexity(int m) {
        ensureValidM(m);
        hyperparameters.put(HP_M, m);
        featureExtractor = basisFunctionGenerator.generate(m);
        linearCombination = LinearCombination.create(m);
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return hyperparameters;
    }

    @Override
    public void setHyperparameter(String name, Object value) {
        switch (name) {
            case HP_M:
                if (value instanceof Integer i) {
                    setModelComplexity(i);
                } else {
                    throw new IllegalArgumentException("Invalid value: " + value);
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported hyperparameter: " + name);
        }
    }

    @Override
    public float[] getParameters() {
        return linearCombination.getParameters();
    }

    @Override
    public float estimateUnary(float input) {
        float[] features = featureExtractor.extractFeatures(input);
        return linearCombination.estimateScalar(features);
    }

    @Override
    public float[] calculateParameterGradients(float[] inputs) {
        return linearCombination.calculateParameterGradients(inputs);
    }

}
