package net.tvburger.jdl.linear.basis;

import net.tvburger.jdl.common.patterns.Strategy;

import java.util.List;

@Strategy(Strategy.Role.INTERFACE)
public interface BasisFunction {

    float apply(float input);

    interface Generator {

        FeatureExtractor generate(int featureCount);

    }

    @Strategy(Strategy.Role.CONCRETE)
    class FeatureExtractor implements net.tvburger.jdl.linear.FeatureExtractor {

        private final List<BasisFunction> basis;

        public FeatureExtractor(List<BasisFunction> basis) {
            this.basis = basis;
        }

        /**
         * {@inheritDoc}
         */
        public float[] extractFeatures(float input) {
            int m = basis.size();
            float[] features = new float[m];
            for (int i = 0; i < m; i++) {
                features[i] = basis.get(i).apply(input);
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
