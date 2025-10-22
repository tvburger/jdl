package net.tvburger.jdl.common.numbers;

import java.util.Arrays;
import java.util.Comparator;

public final class DoubleSupport implements JavaNumberTypeSupport<Double> {

    protected DoubleSupport() {
    }

    @Override
    public String name() {
        return "Double Precision";
    }

    @Override
    public Array<Double> createArray(int length) {
        Double[] doubles = new Double[length];
        Arrays.fill(doubles, 0.0);
        return Array.of(doubles);
    }

    @Override
    public Double[][] createArrayOfArrays(int rows, int columns) {
        Double[][] doubles = new Double[rows][columns];
        for (Double[] array : doubles) {
            Arrays.fill(array, 0.0);
        }
        return doubles;
    }

    @Override
    public Double multiply(Double first, Double second) {
        return first * second;
    }

    @Override
    public Double multiply(Double first, int second) {
        return first * second;
    }

    @Override
    public Double divide(Double first, Double second) {
        return first / second;
    }

    @Override
    public Double divide(Double first, int second) {
        return first / second;
    }

    @Override
    public Double add(Double first, Double second) {
        return first + second;
    }

    @Override
    public Double add(Double first, int second) {
        return first + second;
    }

    @Override
    public Double minusOne() {
        return -1.0;
    }

    @Override
    public Double one() {
        return 1.0;
    }

    @Override
    public Double zero() {
        return 0.0;
    }

    @Override
    public boolean equals(Double first, Double second) {
        double diff = first - second;
        return -1.0e-10 <= diff && diff <= 1.0e-10;
    }

    @Override
    public boolean isGreaterThan(Double first, Double second) {
        return first > second;
    }

    @Override
    public Double squareRoot(Double value) {
        return Math.sqrt(value);
    }

    @Override
    public Double absolute(Double value) {
        return Math.abs(value);
    }

    @Override
    public Double valueOf(double value) {
        return value;
    }

    @Override
    public Comparator<Double> comparator() {
        return Double::compare;
    }

    @Override
    public Double clamp01(Double value) {
        if (value <= zero()) {
            return zero();
        }
        if (value >= one()) {
            return one();
        }
        return value;
    }

    @Override
    public Double epsilon() {
        return 1e-16;
    }

    @Override
    public Double log(Double value) {
        return Math.log(value);
    }

    @Override
    public Double pow(Double base, int times) {
        return Math.pow(base, times);
    }

    @Override
    public boolean isInstance(Object value) {
        return value instanceof Double;
    }

    @Override
    public int compare(Double o1, Double o2) {
        return Double.compare(o1, o2);
    }
}
