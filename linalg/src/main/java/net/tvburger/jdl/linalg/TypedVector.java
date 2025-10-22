package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;

public class TypedVector<N extends Number> implements Vector<N> {

    private final Array<N> values;
    private final boolean columnVector;
    private final JavaNumberTypeSupport<N> typeSupport;

    public TypedVector(Array<N> values, boolean columnVector, JavaNumberTypeSupport<N> typeSupport) {
        this.values = values;
        this.columnVector = columnVector;
        this.typeSupport = typeSupport;
    }

    @Override
    public int getDimensions() {
        return values.length();
    }

    @Override
    public boolean isColumnVector() {
        return columnVector;
    }

    @Override
    public TypedVector<N> transpose() {
        return new TypedVector<>(values, !columnVector, typeSupport);
    }

    @Override
    public TypedMatrix<N> asMatrix() {
        N[][] matrixValues;
        if (columnVector) {
            matrixValues = typeSupport.createArrayOfArrays(values.length(), 1);
            for (int i = 0; i < values.length(); i++) {
                matrixValues[i][0] = values.get(i);
            }
        } else {
            matrixValues = typeSupport.createArrayOfArrays(1, values.length());
            for (int i = 0; i < values.length(); i++) {
                matrixValues[0][i] = values.get(i);
            }
        }
        return new TypedMatrix<>(matrixValues, typeSupport);
    }

    @Override
    public Array<N> asArray() {
        return values;
    }

    @Override
    public N get(int i) {
        return values.get(i - 1);
    }

    @Override
    public TypedVector<N> multiply(N value) {
        return new TypedVector<>(values.clone().apply(n -> typeSupport.multiply(n, value)), columnVector, typeSupport);
    }

    @Override
    public Vector<N> divide(N value) {
        return new TypedVector<>(values.clone().apply(n -> typeSupport.divide(n, value)), columnVector, typeSupport);
    }

    @Override
    public TypedVector<N> add(N value) {
        return new TypedVector<>(values.clone().apply(n -> typeSupport.add(n, value)), columnVector, typeSupport);
    }

    @Override
    public TypedVector<N> subtract(N value) {
        return add(typeSupport.negate(value));
    }

    @Override
    public TypedVector<N> add(Vector<N> vector) {
        Array<N> addedValues = values.clone();
        for (int i = 0; i < values.length(); i++) {
            addedValues.set(i, typeSupport.add(values.get(i), vector.get(i + 1)));
        }
        return new TypedVector<>(addedValues, columnVector, typeSupport);
    }

    @Override
    public TypedVector<N> subtract(Vector<N> vector) {
        return add(vector.multiply(typeSupport.minusOne()));
    }

    @Override
    public N dotProduct(Vector<N> vector) {
        N result = typeSupport.zero();
        if (vector.getDimensions() != getDimensions()) {
            throw new IllegalArgumentException();
        }
        for (int i = 0; i < values.length(); i++) {
            result = typeSupport.add(result, typeSupport.multiply(values.get(i), vector.get(i + 1)));
        }
        return result;
    }

    @Override
    public N norm() {
        N result = typeSupport.zero();
        for (N value : values) {
            result = typeSupport.add(result, typeSupport.multiply(value, value));
        }
        return typeSupport.squareRoot(result);
    }

    @Override
    public int hashCode() {
        return values.hashCode() * (columnVector ? -1 : 1);
    }

    @SuppressWarnings("unchecked")
    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof TypedVector<?> v) {
            return columnVector == v.columnVector && Array.equals(values, (Array<N>) v.values, typeSupport);
        } else {
            return false;
        }
    }

    @Override
    public void print() {
        print(null);
    }

    @Override
    public void print(String name) {
        asMatrix().print(name);
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }
}
