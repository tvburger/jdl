package net.tvburger.jdl.common.numbers;

import java.util.Arrays;

public final class DoubleSupport implements JavaNumberTypeSupport<Double> {

    protected DoubleSupport() {
    }

    @Override
    public String name() {
        return "Double Precision";
    }

    @Override
    public Double[] createArray(int length) {
        Double[] doubles = new Double[length];
        Arrays.fill(doubles, 0.0);
        return doubles;
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
    public Double divide(Double first, Double second) {
        return first / second;
    }

    @Override
    public Double add(Double first, Double second) {
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
    public boolean greaterThan(Double first, Double second) {
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
}
