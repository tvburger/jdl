package net.tvburger.jdl.knn;

public class UniformWeighting implements NeighborWeighting {

    @Override
    public float weight(float distance) {
        return 1;
    }

}
