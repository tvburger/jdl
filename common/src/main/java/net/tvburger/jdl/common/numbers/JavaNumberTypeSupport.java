package net.tvburger.jdl.common.numbers;

import java.math.BigInteger;
import java.util.Comparator;

public interface JavaNumberTypeSupport<N> extends Comparator<N> {

    JavaNumberTypeSupport<Rational<BigInteger>> RATIONAL_BIGINT = new RationalBigIntegerSupport();
    JavaNumberTypeSupport<Rational<Long>> RATIONAL_LONG = new RationalLongSupport();
    JavaNumberTypeSupport<Rational<Integer>> RATIONAL_INT = new RationalIntegerSupport();
    JavaNumberTypeSupport<Double> DOUBLE = new DoubleSupport();
    JavaNumberTypeSupport<Float> FLOAT = new FloatSupport();

    String name();

    Array<N> createArray(int length);

    N[][] createArrayOfArrays(int rows, int columns);

    N multiply(N first, N second);

    N multiply(N first, int second);

    N divide(N first, N second);

    N divide(N first, int second);

    N add(N first, N second);

    N add(N first, int second);

    default N subtract(N first, N second) {
        return add(first, multiply(minusOne(), second));
    }

    default N subtract(N first, int second) {
        return add(first, -second);
    }

    default N negate(N value) {
        return multiply(minusOne(), value);
    }

    N minusOne();

    N one();

    N zero();

    default N inverse(N value) {
        return divide(one(), value);
    }

    default boolean isZero(N value) {
        return equals(zero(), value);
    }

    default boolean isOne(N value) {
        return equals(one(), value);
    }

    boolean equals(N first, N second);

    boolean isGreaterThan(N first, N second);

    default boolean isPositive(N value) {
        return isGreaterThan(value, zero());
    }

    default boolean isNegative(N value) {
        return isGreaterThan(zero(), value);
    }

    default boolean hasSameSign(N first, N second) {
        return isZero(first) || isZero(second)
                || isPositive(first) && isPositive(second)
                || isNegative(first) && isNegative(second);
    }

    N squareRoot(N value);

    default N absolute(N value) {
        return isGreaterThan(zero(), value) ? multiply(minusOne(), value) : value;
    }

    N valueOf(double value);

    Comparator<N> comparator();

    default N max(N first, N second) {
        return isGreaterThan(second, first) ? second : first;
    }

    default N min(N first, N second) {
        return isGreaterThan(first, second) ? second : first;
    }

    N clamp01(N value);

    N epsilon();

    N log(N value);

    default N pow(N base, int times) {
        for (int i = 1; i < times; i++) {
            base = multiply(base, base);
        }
        return base;
    }

    boolean isInstance(Object value);

    @SuppressWarnings("unchecked")
    default N cast(Object value) {
        if (isInstance(value)) {
            return (N) value;
        } else {
            throw new IllegalArgumentException("Cannot cast value to target number type: " + value);
        }
    }

    @Override
    default int compare(N first, N second) {
        if (isGreaterThan(first, second)) {
            return 1;
        } else if (isGreaterThan(second, first)) {
            return -1;
        } else {
            return 0;
        }
    }
}
