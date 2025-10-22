package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;

import java.util.function.BiFunction;

public final class Vectors {

    private Vectors() {
    }

    @SafeVarargs
    public static <N extends Number> TypedVector<N> of(JavaNumberTypeSupport<N> typeSupport, N... values) {
        Array<N> numbers = typeSupport.createArray(values.length);
        for (int i = 0; i < numbers.length(); i++) {
            numbers.set(i, values[i]);
        }
        return new TypedVector<>(numbers, false, typeSupport);
    }

    public static <N extends Number> TypedVector<N> zeros(JavaNumberTypeSupport<N> typeSupport, int dimensions) {
        Array<N> numbers = typeSupport.createArray(dimensions);
        return new TypedVector<>(numbers, false, typeSupport);
    }

    public static TypedVector<Float> of(float... values) {
        Float[] floats = new Float[values.length];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = values[i];
        }
        return new TypedVector<>(Array.of(floats), false, JavaNumberTypeSupport.FLOAT);
    }

    public static TypedVector<Double> of(double... values) {
        Double[] doubles = new Double[values.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = values[i];
        }
        return new TypedVector<>(Array.of(doubles), false, JavaNumberTypeSupport.DOUBLE);
    }

    public static <N extends Number, M extends Number> TypedVector<M> convert(Vector<N> vector, JavaNumberTypeSupport<M> typeSupport) {
        Array<M> numbers = typeSupport.createArray(vector.getDimensions());
        for (int i = 0; i < vector.getDimensions(); i++) {
            numbers.set(i, typeSupport.valueOf(vector.get(i + 1).doubleValue()));
        }
        return new TypedVector<>(numbers, vector.isColumnVector(), typeSupport);
    }

    public static <N extends Number> TypedVector<N> squared(TypedVector<N> vector) {
        return new TypedVector<>(vector.asArray().clone().apply(v -> vector.getNumberTypeSupport().multiply(v, v)), vector.isColumnVector(), vector.getNumberTypeSupport());
    }

    public static <N extends Number> TypedVector<N> squareRoot(TypedVector<N> vector) {
        return new TypedVector<>(vector.asArray().clone().apply(v -> vector.getNumberTypeSupport().squareRoot(v)), vector.isColumnVector(), vector.getNumberTypeSupport());
    }

    public static <N extends Number> TypedVector<N> divide(TypedVector<N> vector, TypedVector<N> denominator) {
        Array<N> values = vector.getNumberTypeSupport().createArray(vector.getDimensions());
        for (int i = 0; i < values.length(); i++) {
            values.set(i, vector.getNumberTypeSupport().divide(vector.get(i + 1), denominator.get(i + 1)));
        }
        return new TypedVector<>(values, vector.isColumnVector(), vector.getNumberTypeSupport());
    }

    public static <N extends Number> TypedVector<N> elementWise(TypedVector<N> vector, BiFunction<Integer, N, N> function) {
        Array<N> values = vector.getNumberTypeSupport().createArray(vector.getDimensions());
        for (int i = 0; i < values.length(); i++) {
            values.set(i, function.apply(i + 1, vector.get(i + 1)));
        }
        return new TypedVector<>(values, vector.isColumnVector(), vector.getNumberTypeSupport());
    }
}
