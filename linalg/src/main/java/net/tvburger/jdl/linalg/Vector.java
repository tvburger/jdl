package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.Tensor;

public interface Vector<N extends Number> extends Transposable<Vector<N>>, Tensor<N> {

    int getDimensions();

    default boolean isRowVector() {
        return !isColumnVector();
    }

    boolean isColumnVector();

    Matrix<N> asMatrix();

    Array<N> asArray();

    N get(int i);

    Vector<N> multiply(N value);

    Vector<N> divide(N value);

    Vector<N> add(N value);

    Vector<N> subtract(N value);

    Vector<N> add(Vector<N> vector);

    Vector<N> subtract(Vector<N> vector);

    N dotProduct(Vector<N> vector);

    N norm();

    @Override
    int hashCode();

    @Override
    boolean equals(Object obj);

    default void print() {
        print(null);
    }

    default void print(String name) {
        asMatrix().print(name);
    }
}
