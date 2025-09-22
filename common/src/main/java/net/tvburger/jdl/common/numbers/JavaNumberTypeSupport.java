package net.tvburger.jdl.common.numbers;

import java.math.BigInteger;

public interface JavaNumberTypeSupport<N> {

    JavaNumberTypeSupport<Rational<BigInteger>> RATIONAL_BIGINT = new RationalBigIntegerSupport();
    JavaNumberTypeSupport<Rational<Long>> RATIONAL_LONG = new RationalLongSupport();
    JavaNumberTypeSupport<Rational<Integer>> RATIONAL_INT = new RationalIntegerSupport();
    JavaNumberTypeSupport<Double> DOUBLE = new DoubleSupport();
    JavaNumberTypeSupport<Float> FLOAT = new FloatSupport();

    String name();

    N[] createArray(int length);

    N[][] createArrayOfArrays(int rows, int columns);

    N multiply(N first, N second);

    N divide(N first, N second);

    N add(N first, N second);

    default N substract(N first, N second) {
        return add(first, multiply(minusOne(), second));
    }

    default N negate(N value) {
        return multiply(minusOne(), value);
    }

    N minusOne();

    N one();

    N zero();

    default N inverse(N value) {
        return multiply(minusOne(), value);
    }

    default boolean isZero(N value) {
        return equals(zero(), value);
    }

    default boolean isOne(N value) {
        return equals(one(), value);
    }

    boolean equals(N first, N second);

    boolean greaterThan(N first, N second);

    N squareRoot(N value);

    default N absolute(N value) {
        return greaterThan(zero(), value) ? multiply(minusOne(), value) : value;
    }

    N valueOf(double value);

}
