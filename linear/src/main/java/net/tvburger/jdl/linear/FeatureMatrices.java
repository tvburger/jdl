package net.tvburger.jdl.linear;

import net.tvburger.jdl.linalg.Matrices;
import net.tvburger.jdl.linalg.TypedMatrix;
import net.tvburger.jdl.model.DataSet;

public final class FeatureMatrices {

    private FeatureMatrices() {
    }

    public static TypedMatrix<Float> create(FeatureExtractor featureExtractor, DataSet dataSet) {
        float[][] cells = new float[dataSet.size()][featureExtractor.featureCount() + 1];
        for (int i = 0; i < cells.length; i++) {
            float x = dataSet.samples().get(i).features()[0];
            float[] features = featureExtractor.extractFeatures(x);
            cells[i][0] = 1;
            for (int j = 0; j < features.length; j++) {
                cells[i][j + 1] = features[j];
            }
        }
        return Matrices.of(cells);
    }

}
