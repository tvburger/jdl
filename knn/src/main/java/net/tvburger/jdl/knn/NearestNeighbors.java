package net.tvburger.jdl.knn;

import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.SimpleHolder;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.distances.DistanceMetric;

import java.util.Comparator;
import java.util.Set;
import java.util.TreeSet;

public class NearestNeighbors implements EstimationFunction {

    private final DistanceMetric distanceMetric;
    private final NeighborWeighting neighborWeighting;
    private DataSet memory;
    private int k;

    public NearestNeighbors(int k, DistanceMetric distanceMetric, NeighborWeighting neighborWeighting) {
        this.distanceMetric = distanceMetric;
        this.neighborWeighting = neighborWeighting;
        this.k = k;
    }

    public DataSet getMemory() {
        return memory;
    }

    public void setMemory(DataSet memory) {
        this.memory = memory;
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive!");
        }
        this.k = k;
    }

    @Override
    public float[] estimate(float[] inputs) {
        if (k < 0) {
            throw new IllegalStateException("Invalid k set (" + k + ")! Must be positive!");
        }
        if (memory == null || memory.samples().isEmpty()) {
            throw new IllegalStateException("Model has not been trained!");
        }
        if (k > memory.samples().size()) {
            throw new IllegalArgumentException("We have not enough samples!");
        }
        float maxDistance = Float.POSITIVE_INFINITY;
        Set<Pair<DataSet.Sample, Float>> neighbors = new TreeSet<>(furthestFirst());
        for (DataSet.Sample sample : memory) {
            float sampleDistance = distanceMetric.distance(sample.features(), inputs);
            if (neighbors.size() < k) {
                neighbors.add(Pair.of(sample, sampleDistance));
                if (sampleDistance < maxDistance) {
                    maxDistance = sampleDistance;
                }
            } else if (sampleDistance < maxDistance) {
                neighbors.remove(neighbors.iterator().next());
                neighbors.add(Pair.of(sample, sampleDistance));
                maxDistance = sampleDistance;
            }
        }
        SimpleHolder<Float> totalWeights = new SimpleHolder<>(0.0f);
        float[] estimation = new float[neighbors.iterator().next().left().targetOutputs().length];
        neighbors.forEach(p -> {
            float distance = p.right();
            float weight = neighborWeighting.weight(distance);
            float[] neighborOutputs = p.left().targetOutputs();
            totalWeights.adjust(f -> f + weight);
            for (int i = 0; i < estimation.length; i++) {
                estimation[i] += weight * neighborOutputs[i];
            }
        });
        for (int i = 0; i < estimation.length; i++) {
            estimation[i] = estimation[i] / totalWeights.get();
        }
        return estimation;
    }

    @Override
    public int arity() {
        if (memory == null || memory.samples().isEmpty()) {
            throw new IllegalStateException("Not trained!");
        }
        return memory.samples().iterator().next().features().length;
    }

    @Override
    public int coArity() {
        if (memory == null || memory.samples().isEmpty()) {
            throw new IllegalStateException("Not trained!");
        }
        return memory.samples().iterator().next().targetOutputs().length;
    }

    private Comparator<? super Pair<DataSet.Sample, Float>> furthestFirst() {
        return (p1, p2) ->
                Floats.equals(p1.right(), p2.right())
                        ? 0
                        : Floats.greaterThan(p1.right(), p2.right())
                        ? -1
                        : 1;
    }
}
