package net.tvburger.jdl.linalg;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class VectorTest {

    @Test
    public void testOf() {
        // When
        Vector<Float> result = Vectors.of(1f, 2f, 3f, 4f);

        // Then
        Assertions.assertEquals(4, result.getDimensions());
        Assertions.assertTrue(result.isRowVector());
        Assertions.assertFalse(result.isColumnVector());
        Assertions.assertEquals(1f, result.get(1));
        Assertions.assertEquals(2f, result.get(2));
        Assertions.assertEquals(3f, result.get(3));
        Assertions.assertEquals(4f, result.get(4));
    }

    @Test
    public void testTranslate() {
        // Given
        Vector<Float> vector = Vectors.of(1f, 2f, 3f, 4f);

        // When
        Vector<Float> result = vector.transpose();

        // Then
        Assertions.assertEquals(4, result.getDimensions());
        Assertions.assertFalse(result.isRowVector());
        Assertions.assertTrue(result.isColumnVector());
        Assertions.assertEquals(1f, result.get(1));
        Assertions.assertEquals(2f, result.get(2));
        Assertions.assertEquals(3f, result.get(3));
        Assertions.assertEquals(4f, result.get(4));
    }

    @Test
    public void testAdd_scalar() {
        // Given
        Vector<Float> vector = Vectors.of(1f, 2f, 3f, 4f);

        // When
        Vector<Float> result = vector.add(4f);

        // Then
        Assertions.assertEquals(5f, result.get(1));
        Assertions.assertEquals(6f, result.get(2));
        Assertions.assertEquals(7f, result.get(3));
        Assertions.assertEquals(8f, result.get(4));
    }

    @Test
    public void testAdd_vector() {
        // Given
        Vector<Float> vectorA = Vectors.of(1f, 2f, 3f, 4f);
        Vector<Float> vectorB = Vectors.of(4f, 3f, 2f, 1f);

        // When
        Vector<Float> result = vectorA.add(vectorB);

        // Then
        Assertions.assertEquals(5f, result.get(1));
        Assertions.assertEquals(5f, result.get(2));
        Assertions.assertEquals(5f, result.get(3));
        Assertions.assertEquals(5f, result.get(4));
    }

    @Test
    public void testMultiply() {
        // Given
        Vector<Float> vector = Vectors.of(1f, 2f, 3f, 4f);

        // When
        Vector<Float> result = vector.multiply(4f);

        // Then
        Assertions.assertEquals(4f, result.get(1));
        Assertions.assertEquals(8f, result.get(2));
        Assertions.assertEquals(12f, result.get(3));
        Assertions.assertEquals(16f, result.get(4));
    }

    @Test
    public void testDotProduct_perpendicular() {
        // Given
        Vector<Float> vectorA = Vectors.of(1f, 0f);
        Vector<Float> vectorB = Vectors.of(0f, 1f);

        // When
        float result = vectorA.dotProduct(vectorB);

        // Then
        Assertions.assertEquals(0.0f, result);
    }

    @Test
    public void testDotProduct_nonPerpendicular() {
        // Given
        Vector<Float> vectorA = Vectors.of(1f, 1f);
        Vector<Float> vectorB = Vectors.of(-3f, 2f);

        // When
        float result = vectorA.dotProduct(vectorB);

        // Then
        Assertions.assertEquals(-1f, result);
    }

    @Test
    public void testNorm() {
        // Given
        Vector<Float> vector = Vectors.of(3f, 4f);

        // When
        float result = vector.norm();

        // Then
        Assertions.assertEquals(5f, result);
    }

    @Test
    public void testAsMatrix() {
        // Given
        Vector<Float> vector = Vectors.of(3f, 4f);

        // When
        Matrix<Float> result = vector.asMatrix();

        // Then
        Assertions.assertEquals(1, result.m());
        Assertions.assertEquals(2, result.n());
        Assertions.assertEquals(3f, result.get(1, 1));
        Assertions.assertEquals(4f, result.get(1, 2));
    }

    @Test
    public void testAsMatrix_transposed() {
        // Given
        Vector<Float> vector = Vectors.of(3f, 4f).transpose();

        // When
        Matrix<Float> result = vector.asMatrix();

        // Then
        Assertions.assertEquals(2, result.m());
        Assertions.assertEquals(1, result.n());
        Assertions.assertEquals(3f, result.get(1, 1));
        Assertions.assertEquals(4f, result.get(2, 1));
    }

    @Test
    public void testPrint() {
        // Given
        Vector<Float> vector = Vectors.of(3f, 4f).transpose();

        // When
        vector.print("t");
        vector.transpose().print("t^T");
    }
}
