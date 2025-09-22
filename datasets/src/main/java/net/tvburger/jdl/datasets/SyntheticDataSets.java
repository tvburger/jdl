package net.tvburger.jdl.datasets;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.scalars.UnaryEstimationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SyntheticDataSets {

    public static abstract class SyntheticDataSet<N extends Number> implements NumberTypeAgnostic<N> {

        private final JavaNumberTypeSupport<N> typeSupport;
        private final Random random = new Random();
        private float noiseScale = 0.1f;
        private float bias = 0.0f;

        protected SyntheticDataSet(JavaNumberTypeSupport<N> typeSupport) {
            this.typeSupport = typeSupport;
        }

        @Override
        public JavaNumberTypeSupport<N> getCurrentNumberType() {
            return typeSupport;
        }

        public void setNoiseScale(float noiseScale) {
            this.noiseScale = noiseScale;
        }

        public float getNoiseScale() {
            return noiseScale;
        }

        public float getBias() {
            return bias;
        }

        public void setBias(float bias) {
            this.bias = bias;
        }

        public void setRandomSeed(int seed) {
            random.setSeed(seed);
        }

        public DataSet<N> loadRandomX(float min, float max, int n) {
            List<DataSet.Sample<N>> samples = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                float x = (max - min) * random.nextFloat() + min;
                float y = targetOutputs(x) + noiseScale * (float) random.nextGaussian(0.0, noiseScale) + bias;
                N[] xa = typeSupport.createArray(1);
                xa[0] = typeSupport.valueOf(x);
                N[] ya = typeSupport.createArray(1);
                ya[0] = typeSupport.valueOf(y);
                samples.add(new DataSet.Sample<>(xa, ya));
            }
            return new DataSet<>(samples);
        }

        public DataSet<N> loadEvenX(float min, float max, int n) {
            List<DataSet.Sample<N>> samples = new ArrayList<>();
            N range = typeSupport.substract(typeSupport.valueOf(max), typeSupport.valueOf(min));
            N counter = typeSupport.zero();
            N n_min_1 = typeSupport.valueOf(n - 1);
            for (int i = 0; i < n; i++) {
                N x = typeSupport.add(typeSupport.multiply(typeSupport.divide(range, n_min_1), counter), typeSupport.valueOf(min));
                N y = typeSupport.valueOf(targetOutputs(x.floatValue()) + noiseScale * (float) random.nextGaussian(0.0, noiseScale));
                N[] xa = typeSupport.createArray(1);
                xa[0] = x;
                N[] ya = typeSupport.createArray(1);
                ya[0] = y;
                samples.add(new DataSet.Sample<>(xa, ya));
                counter = typeSupport.add(counter, typeSupport.one());
            }
            return new DataSet<>(samples);
        }

        public DataSet<N> load(int n, float min, float max) {
            return loadEvenX(min, max, n);
        }

        public DataSet<N> load(int n) {
            return loadEvenX(0.0f, 1.0f, n);
        }

        public DataSet<N> load() {
            return loadEvenX(0.0f, 1.0f, 1000);
        }

        protected abstract float targetOutputs(float x);

        public UnaryEstimationFunction<N> getEstimationFunction() {
            return new UnaryEstimationFunction<>() {
                @Override
                public N estimateUnary(N input) {
                    return typeSupport.valueOf(targetOutputs(input.floatValue()));
                }

                @Override
                public JavaNumberTypeSupport<N> getCurrentNumberType() {
                    return typeSupport;
                }
            };
        }
    }

    public static <N extends Number> SyntheticDataSet<N> line(float bias, float weight, JavaNumberTypeSupport<N> typeSupport) {
        return new SyntheticDataSet<>(typeSupport) {
            @Override
            public float targetOutputs(float x) {
                return weight * x + bias;
            }
        };
    }

    public static <N extends Number> SyntheticDataSet<N> sinus(JavaNumberTypeSupport<N> typeSupport) {
        return sinus(1.0f, (float) Math.PI * 2, typeSupport);
    }

    public static <N extends Number> SyntheticDataSet<N> sinus(float scale, float period, JavaNumberTypeSupport<N> typeSupport) {
        return new SyntheticDataSet<>(typeSupport) {
            @Override
            public float targetOutputs(float x) {
                return scale * (float) Math.sin(x * period);
            }
        };
    }

}
