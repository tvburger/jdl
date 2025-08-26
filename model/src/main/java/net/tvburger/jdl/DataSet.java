package net.tvburger.jdl;

import java.util.List;

public record DataSet(List<Sample> samples) {

    public static DataSet of(Sample... sample) {
        return new DataSet(List.of(sample));
    }

    public record Sample(float[] features, float[] targetOutputs) {

    }

    public DataSet subset(int fromIndex, int toIndex) {
        return new DataSet(samples.subList(fromIndex, toIndex));
    }

}
