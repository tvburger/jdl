package net.tvburger.jdl.linear.basis;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

@Strategy(Strategy.Role.INTERFACE)
public interface BasisFunction<N extends Number> extends NumberTypeAgnostic<N> {

    N apply(N input);

    interface Generator<N extends Number> extends NumberTypeAgnostic<N> {

        FeatureExtractor<N> generate(int featureCount);

    }

    @Strategy(Strategy.Role.CONCRETE)
    class FeatureExtractor<N extends Number> implements net.tvburger.jdl.linear.FeatureExtractor<N> {

        private final List<BasisFunction<N>> basis;
        private final JavaNumberTypeSupport<N> typeSupport;

        public FeatureExtractor(List<BasisFunction<N>> basis, JavaNumberTypeSupport<N> typeSupport) {
            this.basis = basis;
            this.typeSupport = typeSupport;
        }

        @Override
        public JavaNumberTypeSupport<N> getTypeSupport() {
            return typeSupport;
        }

        /**
         * {@inheritDoc}
         */
        public Array<N> extractFeatures(N input) {
            int m = basis.size();
            Array<N> features = typeSupport.createArray(m);
            for (int i = 0; i < m; i++) {
                features.set(i, basis.get(i).apply(input));
            }
            return features;
        }

        /**
         * {@inheritDoc}
         */
        public int featureCount() {
            return basis.size();
        }

    }
}
