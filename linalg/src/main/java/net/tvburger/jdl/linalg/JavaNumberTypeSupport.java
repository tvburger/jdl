package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.utils.Floats;

import java.util.Arrays;

public interface JavaNumberTypeSupport<N> {

    JavaNumberTypeSupport<Double> DOUBLE = new JavaNumberTypeSupport<>() {
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
    };

    JavaNumberTypeSupport<Float> FLOAT = new JavaNumberTypeSupport<>() {

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
        public Float divide(Float first, Float second) {
            return first / second;
        }

        @Override
        public Float add(Float first, Float second) {
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
        public boolean greaterThan(Float first, Float second) {
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
    };


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
}
