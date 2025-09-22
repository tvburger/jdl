package net.tvburger.jdl.model.distances;

import net.tvburger.jdl.common.patterns.Strategy;

/**
 * Concrete implementation of the {@link DistanceMetric} strategy that
 * calculates the <b>Euclidean distance</b> between two points in
 * n-dimensional space.
 * <p>
 * Euclidean distance is defined as the square root of the sum of squared
 * differences between corresponding coordinates:
 *
 * <pre>
 * d(p, q) = sqrt( (p₁ - q₁)² + (p₂ - q₂)² + ... + (pₙ - qₙ)² )
 * </pre>
 */
@Strategy(Strategy.Role.CONCRETE)
public class EuclideanDistance implements DistanceMetric {

    /**
     * Calculates the Euclidean distance between two points.
     *
     * @param point1 first point to compare
     * @param point2 second point to compare
     * @return the Euclidean distance between point1 and point2
     * @throws IllegalArgumentException if the points are null or have different dimensions
     */
    @Override
    public float distance(Float[] point1, Float[] point2) {
        if (point1 == null || point2 == null) {
            throw new IllegalArgumentException("Points must not be null.");
        }
        if (point1.length != point2.length) {
            throw new IllegalArgumentException("Points must have the same dimension.");
        }

        float sum = 0.0f;
        for (int i = 0; i < point1.length; i++) {
            float diff = point1[i] - point2[i];
            sum += diff * diff;
        }

        return (float) Math.sqrt(sum);
    }
}