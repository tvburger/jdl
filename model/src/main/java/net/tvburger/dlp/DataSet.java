package net.tvburger.dlp;

import java.util.List;

public record DataSet(List<Sample> samples) {

    record Sample(float[] features, float[] targetOutputs) {

    }

}
