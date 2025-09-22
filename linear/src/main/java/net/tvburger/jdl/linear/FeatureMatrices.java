package net.tvburger.jdl.linear;

import net.tvburger.jdl.linalg.Matrices;
import net.tvburger.jdl.linalg.TypedMatrix;
import net.tvburger.jdl.model.DataSet;

public final class FeatureMatrices {

    private FeatureMatrices() {
    }

    public static <N extends Number> TypedMatrix<N> create(FeatureExtractor<N> featureExtractor, DataSet<N> dataSet) {
        N[][] cells = featureExtractor.getTypeSupport().createArrayOfArrays(dataSet.size(), featureExtractor.featureCount() + 1);
        for (int i = 0; i < cells.length; i++) {
            N x = dataSet.samples().get(i).features()[0];
            N[] features = featureExtractor.extractFeatures(x);
            cells[i][0] = featureExtractor.getTypeSupport().one();
            for (int j = 0; j < features.length; j++) {
                cells[i][j + 1] = features[j];
            }
        }
        return Matrices.create(cells, featureExtractor.getTypeSupport());
    }

}
