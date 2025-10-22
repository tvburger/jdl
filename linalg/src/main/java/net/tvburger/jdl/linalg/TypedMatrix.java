package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;

import java.util.Arrays;

public class TypedMatrix<N extends Number> implements Matrix<N> {

    // rows -> columns
    private final N[][] values;
    private final JavaNumberTypeSupport<N> typeSupport;

    protected TypedMatrix(N[][] values, JavaNumberTypeSupport<N> typeSupport) {
        this.values = values;
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }

    @Override
    public TypedMatrix<N> add(N value) {
        N[][] addedValues = typeSupport.createArrayOfArrays(n(), m());
        for (int i = 0; i < m(); i++) {
            for (int j = 0; j < n(); j++) {
                addedValues[i][j] = typeSupport.add(values[i][j], value);
            }
        }
        return new TypedMatrix<>(addedValues, typeSupport);
    }

    @Override
    public TypedMatrix<N> add(Matrix<N> matrix) {
        N[][] addedValues = typeSupport.createArrayOfArrays(n(), m());
        for (int i = 0; i < m(); i++) {
            for (int j = 0; j < n(); j++) {
                addedValues[i][j] = typeSupport.add(values[i][j], matrix.get(i + 1, j + 1));
            }
        }
        return new TypedMatrix<>(addedValues, typeSupport);
    }

    @Override
    public TypedMatrix<N> substract(N value) {
        return add(typeSupport.negate(value));
    }

    @Override
    public TypedMatrix<N> substract(Matrix<N> matrix) {
        return add(matrix.multiply(typeSupport.minusOne()));
    }

    @Override
    public TypedMatrix<N> multiply(N value) {
        N[][] multipliedValues = typeSupport.createArrayOfArrays(n(), m());
        for (int i = 0; i < m(); i++) {
            for (int j = 0; j < n(); j++) {
                multipliedValues[i][j] = typeSupport.multiply(values[i][j], value);
            }
        }
        return new TypedMatrix<>(multipliedValues, typeSupport);
    }

    @Override
    public TypedMatrix<N> multiply(Matrix<N> matrix) {
        if (matrix.m() != n()) {
            throw new IllegalArgumentException("invalid dimensions of operand: " + matrix.m() + "x" + matrix.n() + "; must have " + n() + " rows!");
        }
        N[][] multipliedValues = typeSupport.createArrayOfArrays(m(), matrix.n());
        for (int i = 0; i < m(); i++) {
            for (int j = 0; j < matrix.n(); j++) {
                for (int n = 0; n < n(); n++) {
                    multipliedValues[i][j] = typeSupport.add(multipliedValues[i][j], typeSupport.multiply(values[i][n], matrix.get(n + 1, j + 1)));
                }
            }
        }
        return new TypedMatrix<>(multipliedValues, typeSupport);
    }

    @Override
    public TypedVector<N> multiply(Vector<N> vector) {
        TypedMatrix<N> multiplied = multiply(vector.asMatrix());
        TypedVector<N> newVector;
        if (multiplied.m() == 1) {
            Array<N> vectorValues = typeSupport.createArray(multiplied.n());
            for (int j = 0; j < vectorValues.length(); j++) {
                vectorValues.set(j, multiplied.values[0][j]);
            }
            newVector = new TypedVector<>(vectorValues, false, typeSupport);
        } else {
            Array<N> vectorValues = typeSupport.createArray(multiplied.m());
            for (int i = 0; i < vectorValues.length(); i++) {
                vectorValues.set(i, multiplied.values[i][0]);
            }
            newVector = new TypedVector<>(vectorValues, true, typeSupport);
        }
        return newVector;
    }

    @Override
    public int numberOfRows() {
        return values.length;
    }

    @Override
    public N determinant() {
        if (n() != m()) {
            throw new IllegalStateException("Must be square matrix!");
        }
        int n = values.length;

        // Make a copy (so we don’t destroy the input)
        N[][] a = typeSupport.createArrayOfArrays(n, n);
        for (int i = 0; i < n; i++) {
            a[i] = values[i].clone();
        }

        N det = typeSupport.one();
        int swapCount = 0;

        for (int i = 0; i < n; i++) {
            // Find pivot
            int pivot = i;
            for (int j = i + 1; j < n; j++) {
                if (typeSupport.isGreaterThan(typeSupport.absolute(a[j][i]), typeSupport.absolute(a[pivot][i]))) {
                    pivot = j;
                }
            }

            // If pivot element is zero, determinant = 0
            if (typeSupport.isZero(a[pivot][i])) {
                return typeSupport.zero();
            }

            // Swap rows if needed
            if (pivot != i) {
                N[] temp = a[i];
                a[i] = a[pivot];
                a[pivot] = temp;
                swapCount++;
            }

            // Eliminate below pivot
            for (int j = i + 1; j < n; j++) {
                N factor = typeSupport.divide(a[j][i], a[i][i]);
                for (int k = i; k < n; k++) {
                    a[j][k] = typeSupport.subtract(a[j][k], typeSupport.multiply(factor, a[i][k]));
                }
            }
        }

        // Multiply diagonal entries
        for (int i = 0; i < n; i++) {
            det = typeSupport.multiply(det, a[i][i]);
        }

        // Adjust sign based on number of row swaps
        if (swapCount % 2 != 0) {
            det = typeSupport.negate(det);
        }

        return det;
    }

    @Override
    public int numberOfColumns() {
        return values[0].length;
    }

    @Override
    public N get(int row, int column) {
        return values[row - 1][column - 1];
    }

    @Override
    public TypedMatrix<N> transpose() {
        N[][] transposed = typeSupport.createArrayOfArrays(n(), m());
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                transposed[j][i] = values[i][j];
            }
        }
        return new TypedMatrix<>(transposed, typeSupport);
    }

    @Override
    public TypedMatrix<N> pseudoInvert() {
        TypedMatrix<N> transposed = transpose();
        if (m() > n()) {
            return transposed.multiply(this).invert().multiply(transposed);
        } else if (n() > m()) {
            return transposed.multiply(this.multiply(transposed).invert());
        } else {
            return invert();
        }
    }

    @Override
    public TypedMatrix<N> invert() {
        int n = values.length;

        // Create augmented matrix [A | I]
        N[][] augmented = typeSupport.createArrayOfArrays(n, 2 * n);
        for (int i = 0; i < n; i++) {
            System.arraycopy(values[i], 0, augmented[i], 0, n);
            augmented[i][i + n] = typeSupport.one(); // Identity part
        }

        // Perform Gaussian elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            N pivot = augmented[i][i];
            if (typeSupport.isZero(pivot)) {
                // Find a row to swap
                int swapRow = i + 1;
                while (swapRow < n && typeSupport.isZero(augmented[swapRow][i])) {
                    swapRow++;
                }
                if (swapRow == n) {
                    print("INVERSE FAILED FOR ");
                    System.out.println("determinant = " + determinant());
                    throw new ArithmeticException("Matrix is singular and cannot be inverted.");
                }
                N[] temp = augmented[i];
                augmented[i] = augmented[swapRow];
                augmented[swapRow] = temp;
                pivot = augmented[i][i];
            }

            // Normalize pivot row
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] = typeSupport.divide(augmented[i][j], pivot);
            }

            // Eliminate other rows
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    N factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] = typeSupport.subtract(augmented[k][j], typeSupport.multiply(factor, augmented[i][j]));
                    }
                }
            }
        }

        // Extract inverse from augmented matrix
        N[][] inverse = typeSupport.createArrayOfArrays(n, n);
        for (int i = 0; i < n; i++) {
            System.arraycopy(augmented[i], n, inverse[i], 0, n);
        }

        return new TypedMatrix<>(inverse, typeSupport);
    }

    @Override
    public void print(String name) {
        String prefix;
        int length;
        if (name == null) {
            prefix = "";
            length = 0;
        } else {
            prefix = name + " = ";
            length = prefix.length();
        }
        for (int i = 0; i < m(); i++) {
            if (i == (m() % 2 == 1 ? m() : m() - 1) / 2) {
                System.out.print(prefix);
            } else {
                for (int n = 0; n < length; n++) {
                    System.out.print(" ");
                }
            }
            String symbol = m() == 1 ? "[" : i == 0 ? "┌" : i == m() - 1 ? "└" : "│";
            System.out.print(symbol + " ");
            for (int j = 0; j < n(); j++) {
//                System.out.printf(" %8.2g", values[i][j].doubleValue());
                System.out.printf(" %s", values[i][j].toString());
            }
            symbol = m() == 1 ? "]" : i == 0 ? "┐" : i == m() - 1 ? "┘" : "│";
            System.out.println(" " + symbol);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TypedMatrix<?> matrix = (TypedMatrix<?>) o;
        return Arrays.deepEquals(values, matrix.values);
    }

    @Override
    public int hashCode() {
        return Arrays.deepHashCode(values);
    }

}
