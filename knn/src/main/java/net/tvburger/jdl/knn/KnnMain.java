package net.tvburger.jdl.knn;

import net.tvburger.jdl.datasets.StraightLineWithNoise;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.distances.Metrics;
import net.tvburger.jdl.model.training.Trainer;

import java.util.Arrays;

public class KnnMain {

    public static void main(String[] args) {
        NearestNeighbors nearestNeighbors = new NearestNeighbors(1, Metrics.euclidean(), new UniformWeighting());

        DataSet dataSet = new StraightLineWithNoise().load();
        DataSet trainingSet = dataSet.subset(10, dataSet.samples().size());
        DataSet validationSet = dataSet.subset(0, 10);
        NearestNeighborsPreparer regime = new NearestNeighborsPreparer();
        Trainer<NearestNeighbors> nearestNeighborsTrainer = Trainer.of(null, null, null, regime);
        nearestNeighborsTrainer.train(nearestNeighbors, trainingSet);

        for (DataSet.Sample sample : validationSet.samples()) {
            float[] estimate = nearestNeighbors.estimate(sample.features());
            System.out.println("real = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate));
        }
    }
}
