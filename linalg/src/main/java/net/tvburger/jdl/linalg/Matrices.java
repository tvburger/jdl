package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;

public final class Matrices {

    private Matrices() {
    }

    public static <N extends Number> TypedMatrix<N> create(N[][] values, JavaNumberTypeSupport<N> typeSupport) {
        return new TypedMatrix<>(values, typeSupport);
    }

    public static TypedMatrix<Double> of(double[]... rows) {
        Double[][] values = new Double[rows.length][];
        int length = -1;
        for (int i = 0; i < rows.length; i++) {
            double[] row = rows[i];
            if (length < 0) {
                length = row.length;
            } else if (length != row.length) {
                throw new IllegalArgumentException();
            }
            values[i] = new Double[length];
            for (int j = 0; j < length; j++) {
                values[i][j] = row[j];
            }
        }
        return new TypedMatrix<>(values, JavaNumberTypeSupport.DOUBLE);
    }

    public static TypedMatrix<Float> of(float[]... rows) {
        Float[][] values = new Float[rows.length][];
        int length = -1;
        for (int i = 0; i < rows.length; i++) {
            float[] row = rows[i];
            if (length < 0) {
                length = row.length;
            } else if (length != row.length) {
                throw new IllegalArgumentException();
            }
            values[i] = new Float[length];
            for (int j = 0; j < length; j++) {
                values[i][j] = row[j];
            }
        }
        return new TypedMatrix<>(values, JavaNumberTypeSupport.FLOAT);
    }

    public static <N extends Number> TypedMatrix<N> identity(int n, JavaNumberTypeSupport<N> support) {
        N[][] values = support.createArrayOfArrays(n, n);
        for (int i = 0; i < n; i++) {
            values[i][i] = support.one();
        }
        return new TypedMatrix<>(values, support);
    }

    public static <N extends Number> boolean isIdentity(TypedMatrix<N> matrix) {
        return isIdentity(matrix, matrix.getNumberTypeSupport());
    }

    public static <N extends Number> boolean isIdentity(Matrix<N> matrix, JavaNumberTypeSupport<N> support) {
        if (matrix.m() != matrix.n()) {
            return false;
        }
        for (int i = 1; i <= matrix.m(); i++) {
            for (int j = 1; j <= matrix.n(); j++) {
                if (i == j && !support.isOne(matrix.get(i, j)) || i != j && !support.isZero(matrix.get(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }

    public static <N extends Number, M extends Number> Matrix<M> convert(Matrix<N> matrix, JavaNumberTypeSupport<M> typeSupport) {
        M[][] numbers = typeSupport.createArrayOfArrays(matrix.m(), matrix.n());
        for (int i = 0; i < matrix.m(); i++) {
            for (int j = 0; j < matrix.n(); j++) {
                numbers[i][j] = typeSupport.valueOf(matrix.get(i + 1, j + 1).doubleValue());
            }
        }
        return new TypedMatrix<>(numbers, typeSupport);
    }

    public static Matrix<Double> withDoublePrecision(Matrix<Float> matrix) {
        Double[][] doubles = JavaNumberTypeSupport.DOUBLE.createArrayOfArrays(matrix.m(), matrix.n());
        for (int i = 0; i < matrix.m(); i++) {
            for (int j = 0; j < matrix.n(); j++) {
                doubles[i][j] = (double) matrix.get(i + 1, j + 1);
            }
        }
        return new TypedMatrix<>(doubles, JavaNumberTypeSupport.DOUBLE);
    }

    public static Matrix<Float> withSinglePrecision(Matrix<Double> matrix) {
        Float[][] floats = JavaNumberTypeSupport.FLOAT.createArrayOfArrays(matrix.m(), matrix.n());
        for (int i = 0; i < matrix.m(); i++) {
            for (int j = 0; j < matrix.n(); j++) {
                floats[i][j] = (float) (double) matrix.get(i + 1, j + 1);
            }
        }
        return new TypedMatrix<>(floats, JavaNumberTypeSupport.FLOAT);
    }
}
