package net.tvburger.jdl.linalg;

public final class Vectors {

    private Vectors() {
    }

    public static TypedVector<Float> of(float... values) {
        Float[] floats = new Float[values.length];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = values[i];
        }
        return new TypedVector<>(floats, false, JavaNumberTypeSupport.FLOAT);
    }

    public static TypedVector<Double> of(double... values) {
        Double[] doubles = new Double[values.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = values[i];
        }
        return new TypedVector<>(doubles, false, JavaNumberTypeSupport.DOUBLE);
    }

    public static TypedVector<Double> withDoublePrecision(TypedVector<Float> vector) {
        Double[] doubles = new Double[vector.getDimensions()];
        for (int i = 0; i < vector.getDimensions(); i++) {
            doubles[i] = (double) (float) vector.get(i + 1);
        }
        return new TypedVector<>(doubles, vector.isColumnVector(), JavaNumberTypeSupport.DOUBLE);
    }

    public static TypedVector<Float> withSinglePrecision(TypedVector<Double> vector) {
        Float[] floats = new Float[vector.getDimensions()];
        for (int i = 0; i < vector.getDimensions(); i++) {
            floats[i] = (float) (double) vector.get(i + 1);
        }
        return new TypedVector<>(floats, vector.isColumnVector(), JavaNumberTypeSupport.FLOAT);
    }
}
