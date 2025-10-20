package net.tvburger.jdl.datasets;

import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.DataSet;

public final class LogicalDataSets {

    private LogicalDataSets() {
    }

    public static DataSet<Float> toMinusSet(DataSet<Float> dataSet) {
        for (DataSet.Sample<Float> sample : dataSet) {
            for (int i = 0; i < sample.featureCount(); i++) {
                if (Floats.equals(sample.features()[i], 0.0f)) {
                    sample.features()[i] = -1.0f;
                }
            }
            for (int i = 0; i < sample.targetCount(); i++) {
                if (Floats.equals(sample.targetOutputs()[i], 0.0f)) {
                    sample.targetOutputs()[i] = -1.0f;
                }
            }
        }
        return dataSet;
    }

    public static DataSet.Loader<Float> or() {
        return () -> DataSet.of(
                DataSet.Sample.of(Floats.of(0.0f, 0.0f), Floats.of(0.0f)),
                DataSet.Sample.of(Floats.of(0.0f, 1.0f), Floats.of(1.0f)),
                DataSet.Sample.of(Floats.of(1.0f, 0.0f), Floats.of(1.0f)),
                DataSet.Sample.of(Floats.of(1.0f, 1.0f), Floats.of(1.0f)));
    }

    public static DataSet.Loader<Float> and() {
        return () -> DataSet.of(
                DataSet.Sample.of(Floats.of(0.0f, 0.0f), Floats.of(0.0f)),
                DataSet.Sample.of(Floats.of(0.0f, 1.0f), Floats.of(0.0f)),
                DataSet.Sample.of(Floats.of(1.0f, 0.0f), Floats.of(0.0f)),
                DataSet.Sample.of(Floats.of(1.0f, 1.0f), Floats.of(1.0f)));
    }

    public static DataSet.Loader<Float> xor() {
        return () -> DataSet.of(
                DataSet.Sample.of(Floats.of(0.0f, 0.0f), Floats.of(0.0f)),
                DataSet.Sample.of(Floats.of(0.0f, 1.0f), Floats.of(1.0f)),
                DataSet.Sample.of(Floats.of(1.0f, 0.0f), Floats.of(1.0f)),
                DataSet.Sample.of(Floats.of(1.0f, 1.0f), Floats.of(0.0f)));
    }

}
