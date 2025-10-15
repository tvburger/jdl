package net.tvburger.jdl.common.numbers;

import java.util.Objects;

public class Rational<N extends Number> extends Number {

    private final N numerator;
    private final N denominator;

    public Rational(N numerator, N denominator) {
        this.numerator = numerator;
        this.denominator = denominator;
    }

    public N numerator() {
        return numerator;
    }

    public N denominator() {
        return denominator;
    }

    @Override
    public int intValue() {
        return (int) doubleValue();
    }

    @Override
    public long longValue() {
        return (long) doubleValue();
    }

    @Override
    public float floatValue() {
        return (float) doubleValue();
//        float value = (float) doubleValue();
//        return value == Float.NaN && numerator.doubleValue() != Double.NaN && denominator.doubleValue() != Double.NaN ? 0.0f : value;
    }

    @Override
    public double doubleValue() {
        return numerator.doubleValue() / denominator.doubleValue();
//        double value = numerator.doubleValue() / denominator.doubleValue();
//        return value == Double.NaN && numerator.doubleValue() != Double.NaN && denominator.doubleValue() != Double.NaN ? 0.0f : value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Rational<?> rational = (Rational<?>) o;
        return Objects.equals(numerator, rational.numerator) && Objects.equals(denominator, rational.denominator);
    }

    @Override
    public int hashCode() {
        return Objects.hash(numerator, denominator);
    }

    @Override
    public String toString() {
        return numerator + "/" + denominator;
    }
}
