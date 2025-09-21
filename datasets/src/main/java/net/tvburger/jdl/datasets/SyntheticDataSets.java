package net.tvburger.jdl.datasets;

import net.tvburger.jdl.model.DataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SyntheticDataSets {

    public static abstract class SyntheticDataSet {

        private final Random random = new Random();
        private float noiseScale = 0.1f;
        private float bias = 0.0f;

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

        public DataSet loadRandomX(float min, float max, int n) {
            List<DataSet.Sample> samples = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                float x = (max - min) * random.nextFloat() + min;
                float y = targetOutputs(x) + noiseScale * (float) random.nextGaussian(0.0, noiseScale) + bias;
                samples.add(new DataSet.Sample(new float[]{x}, new float[]{y}));
            }
            return new DataSet(samples);
        }

        public DataSet loadEvenX(float min, float max, int n) {
            List<DataSet.Sample> samples = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                float x = (max - min) / (n - 1) * i + min;
                float y = targetOutputs(x) + noiseScale * (float) random.nextGaussian(0.0, noiseScale);
                samples.add(new DataSet.Sample(new float[]{x}, new float[]{y}));
            }
            return new DataSet(samples);
        }

        public DataSet load(int n, float min, float max) {
            return loadEvenX(min, max, n);
        }

        public DataSet load(int n) {
            return loadEvenX(0.0f, 1.0f, n);
        }

        public DataSet load() {
            return loadEvenX(0.0f, 1.0f, 1000);
        }

        public abstract float targetOutputs(float x);

        public float[] targetOutputs(float[] inputs) {
            return new float[]{targetOutputs(inputs[0])};
        }

    }

    public static SyntheticDataSet line(float bias, float weight) {
        return new SyntheticDataSet() {
            @Override
            public float targetOutputs(float x) {
                return weight * x + bias;
            }
        };
    }

    public static SyntheticDataSet sinus() {
        return sinus(1.0f, (float) Math.PI * 2);
    }

    public static SyntheticDataSet sinus(float scale, float period) {
        return new SyntheticDataSet() {
            @Override
            public float targetOutputs(float x) {
                return scale * (float) Math.sin(x * period);
            }
        };
    }

}
