package net.tvburger.jdl.common.numbers;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Comparator;

public final class RationalBigIntegerSupport implements JavaNumberTypeSupport<Rational<BigInteger>> {

    private static final Rational<BigInteger> ONE = new Rational<>(BigInteger.ONE, BigInteger.ONE);
    private static final Rational<BigInteger> MINUS_ONE = new Rational<>(BigInteger.valueOf(-1), BigInteger.ONE);
    private static final Rational<BigInteger> ZERO = new Rational<>(BigInteger.ZERO, BigInteger.ONE);
    private static final Rational<BigInteger> EPSILON = new Rational<>(BigInteger.ONE, BigInteger.valueOf(1_000_000_000));

    protected RationalBigIntegerSupport() {
    }

    @Override
    public String name() {
        return "Rational with BigInteger";
    }

    @SuppressWarnings("unchecked")
    @Override
    public Array<Rational<BigInteger>> createArray(int length) {
        Rational<BigInteger>[] array = (Rational<BigInteger>[]) new Rational<?>[length];
        Arrays.fill(array, ZERO);
        return Array.of(array);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Rational<BigInteger>[][] createArrayOfArrays(int rows, int columns) {
        Rational<BigInteger>[][] rationals = (Rational<BigInteger>[][]) new Rational<?>[rows][columns];
        for (Rational<BigInteger>[] array : rationals) {
            Arrays.fill(array, ZERO);
        }
        return rationals;
    }

    @Override
    public Rational<BigInteger> multiply(Rational<BigInteger> first, Rational<BigInteger> second) {
        return rational(first.numerator().multiply(second.numerator()),
                first.denominator().multiply(second.denominator()));
    }

    @Override
    public Rational<BigInteger> multiply(Rational<BigInteger> first, int second) {
        return rational(first.numerator().multiply(BigInteger.valueOf(second)), first.denominator());
    }

    @Override
    public Rational<BigInteger> divide(Rational<BigInteger> first, Rational<BigInteger> second) {
        return rational(first.numerator().multiply(second.denominator()),
                first.denominator().multiply(second.numerator()));
    }

    @Override
    public Rational<BigInteger> divide(Rational<BigInteger> first, int second) {
        return rational(first.numerator(), first.denominator().multiply(BigInteger.valueOf(second)));
    }

    @Override
    public Rational<BigInteger> add(Rational<BigInteger> first, Rational<BigInteger> second) {
        if (first.denominator().equals(second.denominator())) {
            return rational(first.numerator().add(second.numerator()), first.denominator());
        }
        if (first.denominator().mod(second.denominator()).equals(BigInteger.ZERO)) {
            BigInteger factor = first.denominator().divide(second.denominator());
            return rational(first.numerator().add(second.numerator().multiply(factor)), first.denominator());
        }
        if (second.denominator().mod(first.denominator()).equals(BigInteger.ZERO)) {
            BigInteger factor = second.denominator().divide(first.denominator());
            return rational(first.numerator().multiply(factor).add(second.numerator()), second.denominator());
        }
        return rational(first.numerator().multiply(second.denominator()).add(second.numerator().multiply(first.denominator())), first.denominator().multiply(second.denominator()));
    }

    @Override
    public Rational<BigInteger> add(Rational<BigInteger> first, int second) {
        return rational(first.numerator().add(first.denominator().multiply(BigInteger.valueOf(second))), first.denominator());
    }

    private Rational<BigInteger> rational(BigInteger numerator, BigInteger denominator) {
        if (numerator.equals(denominator)) {
            return ONE;
        }
        BigInteger gcd = gcd(numerator, denominator);
        if (gcd.compareTo(BigInteger.ONE) > 0) {
            numerator = numerator.divide(gcd);
            denominator = denominator.divide(gcd);
        }
        return new Rational<>(numerator, denominator);
    }

    private static BigInteger gcd(BigInteger a, BigInteger b) {
        a = a.abs();
        b = b.abs();
        while (b.compareTo(BigInteger.ZERO) != 0) {
            BigInteger temp = b;
            b = a.mod(b);
            a = temp;
        }
        return a;
    }

    @Override
    public Rational<BigInteger> minusOne() {
        return MINUS_ONE;
    }

    @Override
    public Rational<BigInteger> one() {
        return ONE;
    }

    @Override
    public Rational<BigInteger> zero() {
        return ZERO;
    }

    @Override
    public boolean equals(Rational<BigInteger> first, Rational<BigInteger> second) {
        return first.numerator().multiply(second.denominator()).compareTo(first.denominator().multiply(second.numerator())) == 0;
    }

    @Override
    public boolean isGreaterThan(Rational<BigInteger> first, Rational<BigInteger> second) {
        return first.numerator().multiply(second.denominator()).compareTo(first.denominator().multiply(second.numerator())) > 0;
    }

    @Override
    public Rational<BigInteger> squareRoot(Rational<BigInteger> value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Rational<BigInteger> valueOf(double value) {
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

        return numerator == denominator ? ONE : rational(BigInteger.valueOf(numerator), BigInteger.valueOf(denominator));
    }

    @Override
    public Comparator<Rational<BigInteger>> comparator() {
        return (o1, o2) -> isGreaterThan(o1, o2) ? 1 : isGreaterThan(o2, o1) ? -1 : 0;
    }

    @Override
    public Rational<BigInteger> clamp01(Rational<BigInteger> value) {
        if (value.numerator().compareTo(BigInteger.ZERO) <= 0) {
            return ZERO;
        }
        if (value.numerator().compareTo(value.denominator()) >= 0) {
            return ONE;
        }
        return value;
    }

    @Override
    public Rational<BigInteger> epsilon() {
        return EPSILON;
    }

    @Override
    public Rational<BigInteger> log(Rational<BigInteger> value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isInstance(Object value) {
        return value instanceof Rational<?> r && r.numerator() instanceof BigInteger && r.denominator() instanceof BigInteger;
    }
}
