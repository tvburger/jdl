package net.tvburger.jdl.model.distances;

import net.tvburger.jdl.common.patterns.StaticUtility;

/**
 * Utility for obtaining (singleton) instances of distance metrics.
 */
@StaticUtility
public final class Metrics {

    private static final EuclideanDistance euclideanDistance = new EuclideanDistance();

    private Metrics() {
    }

    /**
     * Gets the Euclidean metric.
     *
     * @return the Euclidean metric
     */
    public static EuclideanDistance euclidean() {
        return euclideanDistance;
    }

}
