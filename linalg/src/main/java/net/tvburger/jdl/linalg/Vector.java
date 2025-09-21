package net.tvburger.jdl.linalg;

public interface Vector<N extends Number> extends Transposable<Vector<N>> {

    int getDimensions();

    default boolean isRowVector() {
        return !isColumnVector();
    }

    boolean isColumnVector();

    Matrix<N> asMatrix();

    N[] asArray();

    N get(int i);

    Vector<N> multiply(N value);

    Vector<N> add(N value);

    Vector<N> substract(N value);

    Vector<N> add(Vector<N> vector);

    Vector<N> substract(Vector<N> vector);

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
