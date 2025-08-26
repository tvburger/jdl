package net.tvburger.jdl.utils;

import net.tvburger.jdl.DataSet;

public final class Floats {

    private static final float EPSILON = 1e-6f;

    private Floats() {
    }

    public static boolean equals(float a, float b) {
        return Math.abs(a - b) < EPSILON;
    }

    public static float[] a(float... f) {
        return f;
    }

    public static DataSet.Sample s(float[] feature, float[] targetOutput) {
        return new DataSet.Sample(feature, targetOutput);
    }

    public static boolean greaterThan(float f1, float f2) {
        return f1 > f2 && !equals(f1, f2);
    }
}
