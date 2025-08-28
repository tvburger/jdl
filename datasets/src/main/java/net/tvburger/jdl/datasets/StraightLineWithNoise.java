package net.tvburger.jdl.datasets;

import net.tvburger.jdl.model.DataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class StraightLineWithNoise implements DataSet.Loader {

    @Override
    public DataSet load() {
        Random rnd = new Random();
        List<DataSet.Sample> samples = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            float x = rnd.nextFloat() * 100f;   // feature
            float y = 3 * x + 7 + (float) rnd.nextGaussian(); // target with noise
            samples.add(new DataSet.Sample(new float[]{x}, new float[]{y}));
        }
        return new DataSet(samples);
    }

}
