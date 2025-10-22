package net.tvburger.jdl.common.numbers;

import java.util.Arrays;
import java.util.Comparator;

public final class RationalLongSupport implements JavaNumberTypeSupport<Rational<Long>> {

    private static final Rational<Long> ONE = new Rational<>(1L, 1L);
    private static final Rational<Long> MINUS_ONE = new Rational<>(-1L, 1L);
    private static final Rational<Long> ZERO = new Rational<>(0L, 1L);
    private static final Rational<Long> EPSILON = new Rational<>(1L, 10_000_000L);

    protected RationalLongSupport() {
    }

    @Override
    public String name() {
        return "Rational with Long";
    }

    @SuppressWarnings("unchecked")
    @Override
    public Array<Rational<Long>> createArray(int length) {
        Rational<Long>[] array = (Rational<Long>[]) new Rational<?>[length];
        Arrays.fill(array, ZERO);
        return Array.of(array);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Rational<Long>[][] createArrayOfArrays(int rows, int columns) {
        Rational<Long>[][] rationals = (Rational<Long>[][]) new Rational<?>[rows][columns];
        for (Rational<Long>[] array : rationals) {
            Arrays.fill(array, ZERO);
        }
        return rationals;
    }

    @Override
    public Rational<Long> multiply(Rational<Long> first, Rational<Long> second) {
        return rational(first.numerator() * second.numerator(),
                first.denominator() * second.denominator());
    }

    @Override
    public Rational<Long> multiply(Rational<Long> first, int second) {
        return rational(first.numerator() * second, first.denominator());
    }

    @Override
    public Rational<Long> divide(Rational<Long> first, Rational<Long> second) {
        return rational(first.numerator() * second.denominator(),
                first.denominator() * second.numerator());
    }

    @Override
    public Rational<Long> divide(Rational<Long> first, int second) {
        return rational(first.numerator(), first.denominator() * second);
    }

    @Override
    public Rational<Long> add(Rational<Long> first, Rational<Long> second) {
        if (first.denominator().equals(second.denominator())) {
            return rational(first.numerator() + second.numerator(), first.denominator());
        }
        if (first.denominator() % second.denominator() == 0) {
            long factor = first.denominator() / second.denominator();
            return rational(first.numerator() + second.numerator() * factor, first.denominator());
        }
        if (second.denominator() % first.denominator() == 0) {
            long factor = second.denominator() / first.denominator();
            return rational(first.numerator() * factor + second.numerator(), second.denominator());
        }
        return rational(first.numerator() * second.denominator() + second.numerator() * first.denominator(), first.denominator() * second.denominator());
    }

    @Override
    public Rational<Long> add(Rational<Long> first, int second) {
        return rational(first.numerator() + first.denominator() * second, first.denominator());
    }

    private Rational<Long> rational(long numerator, long denominator) {
        if (numerator == denominator) {
            return ONE;
        }
        while (numerator % 10 == 0 && denominator % 10 == 0) {
            numerator = numerator / 10;
            denominator = denominator / 10;
        }
        long gcd = gcd(numerator, denominator);
        if (gcd != 1) {
            numerator = numerator / gcd;
            denominator = denominator / gcd;
        }
        return new Rational<>(numerator, denominator);
    }

    private static long gcd(long a, long b) {
        a = Math.abs(a);
        b = Math.abs(b);
        while (b != 0) {
            long temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    @Override
    public Rational<Long> minusOne() {
        return MINUS_ONE;
    }

    @Override
    public Rational<Long> one() {
        return ONE;
    }

    @Override
    public Rational<Long> zero() {
        return ZERO;
    }

    @Override
    public boolean equals(Rational<Long> first, Rational<Long> second) {
        return first.numerator() * second.denominator() == first.denominator() * second.numerator();
    }

    @Override
    public boolean isGreaterThan(Rational<Long> first, Rational<Long> second) {
        return first.numerator() * second.denominator() > first.denominator() * second.numerator();
    }

    @Override
    public Rational<Long> squareRoot(Rational<Long> value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Rational<Long> valueOf(double value) {
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
        long numerator;
        long denominator;

        if (pos == -1) {
            // No decimal point
            numerator = Long.parseLong(s);
            denominator = 1L;
        } else {
            int digitsAfter = s.length() - pos - 1;
            String digits = s.substring(0, pos) + s.substring(pos + 1);

            numerator = Long.parseLong(digits);
            denominator = (long) Math.pow(10, digitsAfter);
        }

        return numerator == denominator ? ONE : rational(numerator, denominator);
    }

    @Override
    public Comparator<Rational<Long>> comparator() {
        return (o1, o2) -> isGreaterThan(o1, o2) ? 1 : isGreaterThan(o2, o1) ? -1 : 0;
    }

    @Override
    public Rational<Long> clamp01(Rational<Long> value) {
        if (value.numerator() <= 0) {
            return ZERO;
        }
        if (value.numerator() >= value.denominator()) {
            return ONE;
        }
        return value;
    }

    @Override
    public Rational<Long> epsilon() {
        return EPSILON;
    }

    @Override
    public Rational<Long> log(Rational<Long> value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isInstance(Object value) {
        return value instanceof Rational<?> r && r.numerator() instanceof Long && r.denominator() instanceof Long;
    }
}
