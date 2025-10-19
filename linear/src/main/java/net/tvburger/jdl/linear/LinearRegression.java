package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.patterns.Facade;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

@Facade
public class LinearRegression<N extends Number> {

    protected final BasisFunction.Generator<N> basisFunctionGenerator;
    private final DataSet<N> trainSet;
    private final Map<String, DataSet<N>> testSets;
    private final Optimizer<LinearBasisFunctionModel<N>, N> optimizer;
    private final Regime regime;

    public LinearRegression(BasisFunction.Generator<N> basisFunctionGenerator, DataSet<N> trainSet, Map<String, DataSet<N>> testSets, Optimizer<LinearBasisFunctionModel<N>, N> optimizer, Regime regime) {
        this.basisFunctionGenerator = basisFunctionGenerator;
        this.trainSet = trainSet;
        this.testSets = testSets;
        this.optimizer = optimizer;
        this.regime = regime;
    }

    public DataSet<N> getTrainSet() {
        return trainSet;
    }

    public Map<String, DataSet<N>> getTestSets() {
        return testSets;
    }

    public Optimizer<LinearBasisFunctionModel<N>, N> getOptimizer() {
        return optimizer;
    }

    public LinearBasisFunctionModel<N> fitComplexity(int m) {
        return fitComplexity(m, Set.of());
    }

    public LinearBasisFunctionModel<N> fitComplexity(int m, Set<ExplicitRegularization<N>> regularizations) {
        LinearBasisFunctionModel<N> regression = LinearBasisFunctionModel.create(m, basisFunctionGenerator);
        ObjectiveFunction<N> objective = Objectives.mSE(basisFunctionGenerator.getCurrentNumberType());
        if (null != regularizations) {
            regularizations.forEach(objective::addRegularization);
        }
        regime.train(regression, trainSet, objective, optimizer);
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

    public float calculateRME(DataSet<N> dataSet, LinearBasisFunctionModel<N> regression) {
        float mse = 0.0f;
        for (DataSet.Sample<N> sample : dataSet) {
            float estimated = regression.estimateScalar(sample.features()).floatValue();
            float target = sample.targetOutputs()[0].floatValue();
            mse += (float) Math.pow(estimated - target, 2) / dataSet.size();
        }
        return (float) Math.sqrt(mse);
    }
}
