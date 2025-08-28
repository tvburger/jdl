package net.tvburger.jdl.knn;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.learning.Trainer;

public class NearestNeighborsTrainer implements Trainer<NearestNeighbors> {

    @Override
    public void train(NearestNeighbors nearestNeighbors, DataSet trainingSet) {
        nearestNeighbors.setMemory(trainingSet);
    }

}
