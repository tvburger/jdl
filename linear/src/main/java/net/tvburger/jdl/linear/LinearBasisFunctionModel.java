package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.HyperparameterConfigurable;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.scalars.TrainableScalarFunction;
import net.tvburger.jdl.model.scalars.UnaryEstimationFunction;

import java.util.HashMap;
import java.util.Map;

@Strategy(Strategy.Role.CONCRETE)
public class LinearBasisFunctionModel<N extends Number> implements LinearModel<N>, TrainableScalarFunction<N>, HyperparameterConfigurable, UnaryEstimationFunction<N> {

    public static final String HP_M = "M";

    private final Map<String, Object> hyperparameters;
    private final BasisFunction.Generator<N> basisFunctionGenerator;
    private FeatureExtractor<N> featureExtractor;
    private LinearCombination<N> linearCombination;

    private static void ensureValidM(int m) {
        if (m < 0) {
            throw new IllegalArgumentException("M must be >= 0!");
        }
    }

    public static <N extends Number> LinearBasisFunctionModel<N> create(int m, BasisFunction.Generator<N> basisFunctionGenerator) {
        return new LinearBasisFunctionModel<>(m, basisFunctionGenerator);
    }

    private static Map<String, Object> createHyperparameters(int m) {
        ensureValidM(m);
        Map<String, Object> hyperparameters = new HashMap<>();
        hyperparameters.put(HP_M, m);
        return hyperparameters;
    }

    protected LinearBasisFunctionModel(int m, BasisFunction.Generator<N> basisFunctionGenerator) {
        this(createHyperparameters(m), basisFunctionGenerator, basisFunctionGenerator.generate(m), LinearCombination.create(m, basisFunctionGenerator.getCurrentNumberType()));
    }

    public LinearBasisFunctionModel(Map<String, Object> hyperparameters, BasisFunction.Generator<N> basisFunctionGenerator, FeatureExtractor<N> featureExtractor, LinearCombination<N> linearCombination) {
        this.hyperparameters = hyperparameters;
        this.basisFunctionGenerator = basisFunctionGenerator;
        this.featureExtractor = featureExtractor;
        this.linearCombination = linearCombination;
    }

    public FeatureExtractor<N> getFeatureExtractor() {
        return featureExtractor;
    }

    public int getModelComplexity() {
        return (Integer) hyperparameters.get(HP_M);
    }

    public void setModelComplexity(int m) {
        ensureValidM(m);
        hyperparameters.put(HP_M, m);
        featureExtractor = basisFunctionGenerator.generate(m);
        linearCombination = LinearCombination.create(m, getCurrentNumberType());
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
    public N[] getParameters() {
        return linearCombination.getParameters();
    }

    @Override
    public N estimateUnary(N input) {
        N[] features = featureExtractor.extractFeatures(input);
        return linearCombination.estimateScalar(features);
    }

    @Override
    public N[] calculateParameterGradients(N[] inputs) {
        return linearCombination.calculateParameterGradients(inputs);
    }

    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return linearCombination.getCurrentNumberType();
    }
}
