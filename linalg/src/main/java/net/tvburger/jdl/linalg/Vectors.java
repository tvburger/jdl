package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;

public final class Vectors {

    private Vectors() {
    }

    @SafeVarargs
    public static <N extends Number> TypedVector<N> of(JavaNumberTypeSupport<N> typeSupport, N... values) {
        N[] numbers = typeSupport.createArray(values.length);
        for (int i = 0; i < numbers.length; i++) {
            numbers[i] = values[i];
        }
        return new TypedVector<>(numbers, false, typeSupport);
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

    public static <N extends Number, M extends Number> TypedVector<M> convert(Vector<N> vector, JavaNumberTypeSupport<M> typeSupport) {
        M[] numbers = typeSupport.createArray(vector.getDimensions());
        for (int i = 0; i < vector.getDimensions(); i++) {
            numbers[i] = typeSupport.valueOf(vector.get(i + 1).doubleValue());
        }
        return new TypedVector<>(numbers, vector.isColumnVector(), typeSupport);
    }

    public static TypedVector<Double> withDoublePrecision(TypedVector<Float> vector) {
        Double[] doubles = JavaNumberTypeSupport.DOUBLE.createArray(vector.getDimensions());
        for (int i = 0; i < vector.getDimensions(); i++) {
            doubles[i] = (double) (float) vector.get(i + 1);
        }
        return new TypedVector<>(doubles, vector.isColumnVector(), JavaNumberTypeSupport.DOUBLE);
    }

    public static TypedVector<Float> withSinglePrecision(TypedVector<Double> vector) {
        Float[] floats = JavaNumberTypeSupport.FLOAT.createArray(vector.getDimensions());
        for (int i = 0; i < vector.getDimensions(); i++) {
            floats[i] = (float) (double) vector.get(i + 1);
        }
        return new TypedVector<>(floats, vector.isColumnVector(), JavaNumberTypeSupport.FLOAT);
    }

    public static <N extends Number> TypedVector<N> squared(TypedVector<N> vector) {
        N[] values = vector.getCurrentNumberType().createArray(vector.getDimensions());
        for (int i = 0; i < values.length; i++) {
            N v = vector.get(i + 1);
            values[i] = vector.getCurrentNumberType().multiply(v, v);
        }
        return new TypedVector<>(values, vector.isColumnVector(), vector.getCurrentNumberType());
    }

    public static <N extends Number> TypedVector<N> squareRoot(TypedVector<N> vector) {
        N[] values = vector.getCurrentNumberType().createArray(vector.getDimensions());
        for (int i = 0; i < values.length; i++) {
            N v = vector.get(i + 1);
            values[i] = vector.getCurrentNumberType().squareRoot(v);
        }
        return new TypedVector<>(values, vector.isColumnVector(), vector.getCurrentNumberType());
    }

    public static <N extends Number> TypedVector<N> divide(TypedVector<N> vector, TypedVector<N> denominator) {
        N[] values = vector.getCurrentNumberType().createArray(vector.getDimensions());
        for (int i = 0; i < values.length; i++) {
            values[i] = vector.getCurrentNumberType().divide(vector.get(i + 1), denominator.get(i + 1));
        }
        return new TypedVector<>(values, vector.isColumnVector(), vector.getCurrentNumberType());
    }
}
