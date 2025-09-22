package net.tvburger.jdl.knn;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.distances.Metrics;
import net.tvburger.jdl.model.training.Trainer;

import java.util.Arrays;

public class KnnMain {

    public static void main(String[] args) {
        NearestNeighbors nearestNeighbors = new NearestNeighbors(3, Metrics.euclidean(), new UniformWeighting());

        SyntheticDataSets.SyntheticDataSet<Float> sinus = SyntheticDataSets.sinus(10.0f, (float) Math.PI * 2, JavaNumberTypeSupport.FLOAT);
        DataSet<Float> dataSet = sinus.load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> validationSet = dataSet.subset(0, 10);
        NearestNeighborsPreparer regime = new NearestNeighborsPreparer();
        Trainer<NearestNeighbors> nearestNeighborsTrainer = Trainer.of(null, null, null, regime);
        nearestNeighborsTrainer.train(nearestNeighbors, trainingSet);

        for (DataSet.Sample<Float> sample : validationSet) {
            Float[] estimate = nearestNeighbors.estimate(sample.features());
            System.out.println("with noise = " + Arrays.toString(sample.targetOutputs()) + " estimated = " + Arrays.toString(estimate) + " real = " + Arrays.toString(sinus.getEstimationFunction().estimate(sample.features())));
        }
    }
}
