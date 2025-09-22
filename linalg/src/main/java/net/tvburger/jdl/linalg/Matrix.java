package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;

public interface Matrix<N extends Number> extends Transposable<Matrix<N>>, Invertible<Matrix<N>>, NumberTypeAgnostic<N> {

    Matrix<N> add(N value);

    Matrix<N> add(Matrix<N> matrix);

    Matrix<N> substract(N value);

    Matrix<N> substract(Matrix<N> matrix);

    Matrix<N> multiply(N value);

    Matrix<N> multiply(Matrix<N> matrix);

    Vector<N> multiply(Vector<N> vector);

    int numberOfRows();

    N determinant();

    default int m() {
        return numberOfRows();
    }

    int numberOfColumns();

    default int n() {
        return numberOfColumns();
    }

    N get(int row, int column);

    Matrix<N> pseudoInvert();

    default void print() {
        print(null);
    }

    void print(String name);

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
