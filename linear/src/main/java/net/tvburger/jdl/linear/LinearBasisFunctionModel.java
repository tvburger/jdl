package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.scalars.AffineTransformation;
import net.tvburger.jdl.model.scalars.UnaryEstimationFunction;

@Strategy(Strategy.Role.CONCRETE)
public class LinearBasisFunctionModel<N extends Number> extends AffineTransformation<N> implements LinearModel<N>, UnaryEstimationFunction<N> {

    private final FeatureExtractor<N> featureExtractor;

    private static void ensureValidM(int m) {
        if (m < 0) {
            throw new IllegalArgumentException("M must be >= 0!");
        }
    }

    public static <N extends Number> LinearBasisFunctionModel<N> create(int m, BasisFunction.Generator<N> basisFunctionGenerator) {
        ensureValidM(m);
        return new LinearBasisFunctionModel<>(m, basisFunctionGenerator);
    }

    protected LinearBasisFunctionModel(int modelComplexity, BasisFunction.Generator<N> basisFunctionGenerator) {
        this(basisFunctionGenerator.generate(modelComplexity), basisFunctionGenerator.getCurrentNumberType().zero(), basisFunctionGenerator.getCurrentNumberType().createArray(modelComplexity), basisFunctionGenerator.getCurrentNumberType());
    }

    private LinearBasisFunctionModel(FeatureExtractor<N> featureExtractor, N bias, N[] weights, JavaNumberTypeSupport<N> typeSupport) {
        super(bias, weights, typeSupport);
        this.featureExtractor = featureExtractor;
    }

    public FeatureExtractor<N> getFeatureExtractor() {
        return featureExtractor;
    }

    public int getModelComplexity() {
        return arity();
    }

    @Override
    public N estimateUnary(N input) {
        N[] features = featureExtractor.extractFeatures(input);
        return super.estimateScalar(features);
    }

    @Override
    public N estimateScalar(N[] inputs) {
        if (inputs.length != 1) {
            throw new IllegalArgumentException("Invalid arity!");
        }
        return estimateUnary(inputs[0]);
    }

}