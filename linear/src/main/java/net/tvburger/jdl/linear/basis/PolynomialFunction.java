package net.tvburger.jdl.linear.basis;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.ArrayList;
import java.util.List;

@Strategy(Strategy.Role.CONCRETE)
public class PolynomialFunction<N extends Number> implements BasisFunction<N> {

    private final int m;
    private final JavaNumberTypeSupport<N> typeSupport;

    public PolynomialFunction(int m, JavaNumberTypeSupport<N> typeSupport) {
        this.m = m;
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }

    @Override
    public N apply(N input) {
        N power = typeSupport.one();
        for (int i = 0; i < m; i++) {
            power = typeSupport.multiply(power, input);
        }
        return power;
    }

    public static class Generator<N extends Number> implements BasisFunction.Generator<N> {

        private final JavaNumberTypeSupport<N> typeSupport;

        public Generator(JavaNumberTypeSupport<N> typeSupport) {
            this.typeSupport = typeSupport;
        }

        @Override
        public FeatureExtractor<N> generate(int featureCount) {
            List<BasisFunction<N>> polynomials = new ArrayList<>(featureCount);
            for (int i = 0; i < featureCount; i++) {
                polynomials.add(new PolynomialFunction<>(i + 1, typeSupport));
            }
            return new FeatureExtractor<>(polynomials, typeSupport);
        }

        @Override
        public JavaNumberTypeSupport<N> getNumberTypeSupport() {
            return typeSupport;
        }

    }
}
