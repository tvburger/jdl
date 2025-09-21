package net.tvburger.jdl.linear.basis;

import net.tvburger.jdl.common.patterns.Strategy;

import java.util.ArrayList;
import java.util.List;

@Strategy(Strategy.Role.CONCRETE)
public class PolynomialFunction implements BasisFunction {

    private final int m;

    public PolynomialFunction(int m) {
        this.m = m;
    }

    @Override
    public float apply(float input) {
        return (float) Math.pow(input, m);
    }

    public static class Generator implements BasisFunction.Generator {

        @Override
        public FeatureExtractor generate(int featureCount) {
            List<BasisFunction> polynomials = new ArrayList<>(featureCount);
            for (int i = 0; i < featureCount; i++) {
                polynomials.add(new PolynomialFunction(i + 1));
            }
            return new FeatureExtractor(polynomials);
        }

    }
}
