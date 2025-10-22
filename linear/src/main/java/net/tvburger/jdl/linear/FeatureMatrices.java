package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.patterns.StaticFactory;
import net.tvburger.jdl.common.patterns.StaticUtility;
import net.tvburger.jdl.linalg.Matrices;
import net.tvburger.jdl.linalg.TypedMatrix;
import net.tvburger.jdl.model.DataSet;

@StaticUtility
public final class FeatureMatrices {

    private FeatureMatrices() {
    }

    @StaticFactory
    public static <N extends Number> TypedMatrix<N> create(FeatureExtractor<N> featureExtractor, DataSet<N> dataSet) {
        N[][] cells = featureExtractor.getTypeSupport().createArrayOfArrays(dataSet.size(), featureExtractor.featureCount() + 1);
        for (int i = 0; i < cells.length; i++) {
            N x = dataSet.samples().get(i).features().get(0);
            Array<N> features = featureExtractor.extractFeatures(x);
            cells[i][0] = featureExtractor.getTypeSupport().one();
            for (int j = 0; j < features.length(); j++) {
                cells[i][j + 1] = features.get(j);
            }
        }
        return Matrices.create(cells, featureExtractor.getTypeSupport());
    }

}
