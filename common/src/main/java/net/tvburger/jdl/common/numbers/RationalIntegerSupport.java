package net.tvburger.jdl.common.numbers;

import java.util.Arrays;

public final class RationalIntegerSupport implements JavaNumberTypeSupport<Rational<Integer>> {

    private static final Rational<Integer> ONE = new Rational<>(1, 1);
    private static final Rational<Integer> MINUS_ONE = new Rational<>(-1, 1);
    private static final Rational<Integer> ZERO = new Rational<>(0, 1);

    protected RationalIntegerSupport() {
    }

    @Override
    public String name() {
        return "Rational with Integer";
    }

    @SuppressWarnings("unchecked")
    @Override
    public Rational<Integer>[] createArray(int length) {
        Rational<Integer>[] array = (Rational<Integer>[]) new Rational<?>[length];
        Arrays.fill(array, ZERO);
        return array;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Rational<Integer>[][] createArrayOfArrays(int rows, int columns) {
        Rational<Integer>[][] rationals = (Rational<Integer>[][]) new Rational<?>[rows][columns];
        for (Rational<Integer>[] array : rationals) {
            Arrays.fill(array, ZERO);
        }
        return rationals;
    }

    @Override
    public Rational<Integer> multiply(Rational<Integer> first, Rational<Integer> second) {
        return rational(first.numerator() * second.numerator(),
                first.denominator() * second.denominator());
    }

    @Override
    public Rational<Integer> divide(Rational<Integer> first, Rational<Integer> second) {
        return rational(first.numerator() * second.denominator(),
                first.denominator() * second.numerator());
    }

    @Override
    public Rational<Integer> add(Rational<Integer> first, Rational<Integer> second) {
        if (first.denominator().equals(second.denominator())) {
            return rational(first.numerator() + second.numerator(), first.denominator());
        }
        if (first.denominator() % second.denominator() == 0) {
            int factor = first.denominator() / second.denominator();
            return rational(first.numerator() + second.numerator() * factor, first.denominator());
        }
        if (second.denominator() % first.denominator() == 0) {
            int factor = second.denominator() / first.denominator();
            return rational(first.numerator() * factor + second.numerator(), second.denominator());
        }
        return rational(first.numerator() * second.denominator() + second.numerator() * first.denominator(), first.denominator() * second.denominator());
    }

    private Rational<Integer> rational(int numerator, int denominator) {
        if (numerator == denominator) {
            return ONE;
        }
        while (numerator % 10 == 0 && denominator % 10 == 0) {
            numerator = numerator / 10;
            denominator = denominator / 10;
        }
        int gcd = gcd(numerator, denominator);
        if (gcd != 1) {
            numerator = numerator / gcd;
            denominator = denominator / gcd;
        }
        return new Rational<>(numerator, denominator);
    }

    private static int gcd(int a, int b) {
        a = Math.abs(a);
        b = Math.abs(b);
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    @Override
    public Rational<Integer> minusOne() {
        return MINUS_ONE;
    }

    @Override
    public Rational<Integer> one() {
        return ONE;
    }

    @Override
    public Rational<Integer> zero() {
        return ZERO;
    }

    @Override
    public boolean equals(Rational<Integer> first, Rational<Integer> second) {
        return first.numerator() * second.denominator() == first.denominator() * second.numerator();
    }

    @Override
    public boolean greaterThan(Rational<Integer> first, Rational<Integer> second) {
        return first.numerator() * second.denominator() > first.denominator() * second.numerator();
    }

    @Override
    public Rational<Integer> squareRoot(Rational<Integer> value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Rational<Integer> valueOf(double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("NaN/Infinity not supported");
        }
        if (value == 0.0) return ZERO;
        value = Math.round(value * 1000) / 1000.0;
        String s = Double.toString(value);

        // Handle scientific notation by converting to plain string
        if (s.contains("E") || s.contains("e")) {
            s = new java.math.BigDecimal(s).toPlainString();
        }

        int pos = s.indexOf('.');
        int numerator;
        int denominator;

        if (pos == -1) {
            // No decimal point
            numerator = Integer.parseInt(s);
            denominator = 1;
        } else {
            int digitsAfter = s.length() - pos - 1;
            String digits = s.substring(0, pos) + s.substring(pos + 1);

            numerator = Integer.parseInt(digits);
            denominator = (int) Math.pow(10, digitsAfter);
        }

        return numerator == denominator ? ONE : rational(numerator, denominator);
    }

}
