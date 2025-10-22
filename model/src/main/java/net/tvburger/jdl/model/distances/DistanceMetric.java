package net.tvburger.jdl.model.distances;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;

/**
 * This class represents a specific distance metrics to calculate the distances.
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface DistanceMetric {

    /**
     * Calculates the distance between the 2 points
     *
     * @param point1 point1 to compare
     * @param point2 point2 to compare
     * @return the difference between point1 and point2
     */
    float distance(Array<Float> point1, Array<Float> point2);

}
