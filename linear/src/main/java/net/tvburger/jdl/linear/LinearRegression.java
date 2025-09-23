package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.patterns.Facade;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.optimizer.LinearModelOptimizer;
import net.tvburger.jdl.model.DataSet;

import java.util.LinkedHashMap;
import java.util.Map;

@Facade
public class LinearRegression<N extends Number> {

    protected final BasisFunction.Generator<N> basisFunctionGenerator;
    private final DataSet<N> trainSet;
    private final Map<String, DataSet<N>> testSets;
    private final LinearModelOptimizer<N> optimizer;


    public LinearRegression(BasisFunction.Generator<N> basisFunctionGenerator, DataSet<N> trainSet, Map<String, DataSet<N>> testSets, LinearModelOptimizer<N> optimizer) {
        this.basisFunctionGenerator = basisFunctionGenerator;
        this.trainSet = trainSet;
        this.testSets = testSets;
        this.optimizer = optimizer;
    }

    public DataSet<N> getTrainSet() {
        return trainSet;
    }

    public Map<String, DataSet<N>> getTestSets() {
        return testSets;
    }

    public LinearModelOptimizer<N> getOptimizer() {
        return optimizer;
    }

    public LinearBasisFunctionModel<N> fitComplexity(int m) {
        LinearBasisFunctionModel<N> regression = LinearBasisFunctionModel.create(m, basisFunctionGenerator);
        optimizer.setOptimalWeights(regression, trainSet);
        return regression;
    }

    public Pair<Float, Map<String, Float>> calculateRMEs(LinearBasisFunctionModel<N> model) {
        float trainRme = calculateRME(trainSet, model);
        Map<String, Float> testRmes = new LinkedHashMap<>();
        for (Map.Entry<String, DataSet<N>> testSetEntry : testSets.entrySet()) {
            testRmes.put(testSetEntry.getKey(), calculateRME(testSetEntry.getValue(), model));
        }
        return Pair.of(trainRme, testRmes);
    }

    private float calculateRME(DataSet<N> dataSet, LinearBasisFunctionModel<N> regression) {
        float mse = 0.0f;
        for (DataSet.Sample<N> sample : dataSet) {
            float estimated = regression.estimateScalar(sample.features()).floatValue();
            float target = sample.targetOutputs()[0].floatValue();
            mse += (float) Math.pow(estimated - target, 2) / dataSet.size();
        }
        return (float) Math.sqrt(mse);
    }
}
