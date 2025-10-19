package net.tvburger.jdl.common.numbers;

import net.tvburger.jdl.common.utils.Floats;

import java.util.Arrays;
import java.util.Comparator;

public final class FloatSupport implements JavaNumberTypeSupport<Float> {

    protected FloatSupport() {
    }

    @Override
    public String name() {
        return "Single Precision";
    }

    @Override
    public Float[] createArray(int length) {
        Float[] floats = new Float[length];
        Arrays.fill(floats, 0.0f);
        return floats;
    }

    @Override
    public Float[][] createArrayOfArrays(int rows, int columns) {
        Float[][] floats = new Float[rows][columns];
        for (Float[] array : floats) {
            Arrays.fill(array, 0.0f);
        }
        return floats;
    }

    @Override
    public Float multiply(Float first, Float second) {
        return first * second;
    }

    @Override
    public Float multiply(Float first, int second) {
        return first * second;
    }

    @Override
    public Float divide(Float first, Float second) {
        return first / second;
    }

    @Override
    public Float divide(Float first, int second) {
        return first / second;
    }

    @Override
    public Float add(Float first, Float second) {
        return first + second;
    }

    @Override
    public Float add(Float first, int second) {
        return first + second;
    }

    @Override
    public Float minusOne() {
        return -1.0f;
    }

    @Override
    public Float one() {
        return 1.0f;
    }

    @Override
    public Float zero() {
        return 0.0f;
    }

    @Override
    public boolean equals(Float first, Float second) {
        return Floats.equals(first, second);
    }

    @Override
    public boolean isGreaterThan(Float first, Float second) {
        return first > second;
    }

    @Override
    public Float squareRoot(Float value) {
        return (float) Math.sqrt(value);
    }

    @Override
    public Float absolute(Float value) {
        return Math.abs(value);
    }

    @Override
    public Float valueOf(double value) {
        return (float) value;
    }

    @Override
    public Comparator<Float> comparator() {
        return Float::compare;
    }

    @Override
    public Float clamp01(Float value) {
        if (value <= zero()) {
            return zero();
        }
        if (value >= one()) {
            return one();
        }
        return value;
    }

    @Override
    public Float epsilon() {
        return Floats.EPSILON;
    }

    @Override
    public Float log(Float value) {
        return (float) Math.log(value);
    }

    @Override
    public Float pow(Float base, int times) {
        return (float) Math.pow(base, times);
    }

    @Override
    public boolean isInstance(Object value) {
        return value instanceof Float;
    }
}
