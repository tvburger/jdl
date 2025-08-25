package net.tvburger.dlp.utils;

import net.tvburger.dlp.DataSet;

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
}
