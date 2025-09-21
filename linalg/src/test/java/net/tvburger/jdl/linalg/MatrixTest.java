package net.tvburger.jdl.linalg;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class MatrixTest {

    @Test
    public void testTranspose() {
        // Given
        Matrix<Float> matrix = Matrices.of(new float[]{1.0f, 2.0f}, new float[]{3.0f, 4.0f}, new float[]{5.0f, 6.0f});

        // When
        Matrix<Float> result = matrix.transpose();

        // Then
        Assertions.assertEquals(2, result.m());
        Assertions.assertEquals(3, result.n());
        Assertions.assertEquals(1f, result.get(1, 1));
        Assertions.assertEquals(3f, result.get(1, 2));
        Assertions.assertEquals(5f, result.get(1, 3));
        Assertions.assertEquals(2f, result.get(2, 1));
        Assertions.assertEquals(4f, result.get(2, 2));
        Assertions.assertEquals(6f, result.get(2, 3));
    }

    @Test
    public void testMultiply() {
        // Given
        Matrix<Float> matrixA = Matrices.of(new float[]{1.0f, 2.0f}, new float[]{3.0f, 4.0f}, new float[]{5.0f, 6.0f});
        Matrix<Float> matrixB = Matrices.of(new float[]{1.0f, 2.0f, 3.0f}, new float[]{4.0f, 5.0f, 6.0f});

        // When
        Matrix<Float> result = matrixA.multiply(matrixB);

        // Then
        Assertions.assertEquals(3, result.m());
        Assertions.assertEquals(3, result.n());
        Assertions.assertEquals(9f, result.get(1, 1));
        Assertions.assertEquals(12f, result.get(1, 2));
        Assertions.assertEquals(15f, result.get(1, 3));
        Assertions.assertEquals(19f, result.get(2, 1));
        Assertions.assertEquals(26f, result.get(2, 2));
        Assertions.assertEquals(33f, result.get(2, 3));
        Assertions.assertEquals(29f, result.get(3, 1));
        Assertions.assertEquals(40f, result.get(3, 2));
        Assertions.assertEquals(51f, result.get(3, 3));
    }

    @Test
    public void testInverse() {
        // Given
        Matrix<Float> matrix = Matrices.of(new float[]{1.0f, 2.0f}, new float[]{3.0f, 4.0f});

        // When
        Matrix<Float> result = matrix.invert();

        // Then
        Assertions.assertEquals(2, result.m());
        Assertions.assertEquals(2, result.n());
        Assertions.assertEquals(-2.0f, result.get(1, 1));
        Assertions.assertEquals(1.0f, result.get(1, 2));
        Assertions.assertEquals(1.5f, result.get(2, 1));
        Assertions.assertEquals(-0.5f, result.get(2, 2));
    }

    @Test
    public void testMultiply_inverse() {
        // Given
        Matrix<Float> matrix = Matrices.of(new float[]{1.0f, 2.0f}, new float[]{3.0f, 4.0f});
        Matrix<Float> matrixInverse = matrix.invert();

        // When
        Matrix<Float> result = matrix.multiply(matrixInverse);

        // Then
        Assertions.assertEquals(2, result.m());
        Assertions.assertEquals(2, result.n());
        Assertions.assertEquals(1.0f, result.get(1, 1));
        Assertions.assertEquals(0.0f, result.get(1, 2));
        Assertions.assertEquals(0.0f, result.get(2, 1));
        Assertions.assertEquals(1.0f, result.get(2, 2));
    }

    @Test
    public void testMultiply_vector() {
        // Given
        Matrix<Float> matrix = Matrices.of(new float[]{2.0f, 0.0f}, new float[]{0.0f, 4.0f});
        Vector<Float> vector = Vectors.of(1.0f, 1.0f).transpose();

        // When
        Vector<Float> result = matrix.multiply(vector);

        // Then
        Assertions.assertEquals(2, vector.getDimensions());
        Assertions.assertEquals(2.0f, result.get(1));
        Assertions.assertEquals(4.0f, result.get(2));
    }

    @Test
    public void testPseudoInvert_underdetermined() {
        // Given
        TypedMatrix<Float> matrix = Matrices.of(new float[]{1.0f, 9.0f, 3.0f}, new float[]{1.0f, 5.0f, 7.0f});

        // When
        TypedMatrix<Float> result = matrix.pseudoInvert();
        matrix.print("A");
        result.print("A" + Notations.PSEUDO_INVERSE);
        matrix.multiply(result).print("I");

        // Then
        Assertions.assertTrue(Matrices.isIdentity(matrix.multiply(result)));
    }

    @Test
    public void testPseudoInvert_overdetermined() {
        // Given
        TypedMatrix<Float> matrix = Matrices.of(new float[]{1.0f, 2.0f}, new float[]{3.0f, 1.0f}, new float[]{5.0f, 7.0f});

        // When
        TypedMatrix<Float> result = matrix.pseudoInvert();
        matrix.print("A");
        result.print("A" + Notations.PSEUDO_INVERSE);
        result.multiply(matrix).print("I");

        // Then
        Assertions.assertTrue(Matrices.isIdentity(result.multiply(matrix)));
    }
}
