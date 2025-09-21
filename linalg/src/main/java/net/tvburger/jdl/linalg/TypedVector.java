package net.tvburger.jdl.linalg;

import java.util.Arrays;

public class TypedVector<N extends Number> implements Vector<N> {

    private final N[] values;
    private final boolean columnVector;
    private final JavaNumberTypeSupport<N> typeSupport;

    public TypedVector(N[] values, boolean columnVector, JavaNumberTypeSupport<N> typeSupport) {
        this.values = values;
        this.columnVector = columnVector;
        this.typeSupport = typeSupport;
    }

    @Override
    public int getDimensions() {
        return values.length;
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
            matrixValues = typeSupport.createArrayOfArrays(values.length, 1);
            for (int i = 0; i < values.length; i++) {
                matrixValues[i][0] = values[i];
            }
        } else {
            matrixValues = typeSupport.createArrayOfArrays(1, values.length);
            System.arraycopy(values, 0, matrixValues[0], 0, values.length);
        }
        return new TypedMatrix<>(matrixValues, typeSupport);
    }

    @Override
    public N[] asArray() {
        return values;
    }

    @Override
    public N get(int i) {
        return values[i - 1];
    }

    @Override
    public TypedVector<N> multiply(N value) {
        N[] multipliedValues = typeSupport.createArray(values.length);
        for (int i = 0; i < values.length; i++) {
            multipliedValues[i] = typeSupport.multiply(values[i], value);
        }
        return new TypedVector<>(multipliedValues, columnVector, typeSupport);
    }

    @Override
    public TypedVector<N> add(N value) {
        N[] addedValues = typeSupport.createArray(values.length);
        for (int i = 0; i < values.length; i++) {
            addedValues[i] = typeSupport.add(values[i], value);
        }
        return new TypedVector<>(addedValues, columnVector, typeSupport);
    }

    @Override
    public TypedVector<N> substract(N value) {
        return add(typeSupport.inverse(value));
    }

    @Override
    public TypedVector<N> add(Vector<N> vector) {
        N[] addedValues = typeSupport.createArray(values.length);
        for (int i = 0; i < values.length; i++) {
            addedValues[i] = typeSupport.add(values[i], vector.get(i + 1));
        }
        return new TypedVector<>(addedValues, columnVector, typeSupport);
    }

    @Override
    public TypedVector<N> substract(Vector<N> vector) {
        return add(vector.multiply(typeSupport.minusOne()));
    }

    @Override
    public N dotProduct(Vector<N> vector) {
        N result = typeSupport.zero();
        if (vector.getDimensions() != getDimensions()) {
            throw new IllegalArgumentException();
        }
        for (int i = 0; i < values.length; i++) {
            result = typeSupport.add(result, typeSupport.multiply(values[i], vector.get(i + 1)));
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
        return Arrays.hashCode(values) * (columnVector ? -1 : 1);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof TypedVector v) {
            return columnVector == v.columnVector && Arrays.equals(values, v.values);
        } else {
            return false;
        }
    }

    public void print() {
        print(null);
    }

    public void print(String name) {
        asMatrix().print(name);
    }
}
