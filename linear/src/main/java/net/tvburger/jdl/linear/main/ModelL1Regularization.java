package net.tvburger.jdl.linear.main;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.Threads;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linear.LinearRegression;
import net.tvburger.jdl.linear.basis.BasisFunction;
import net.tvburger.jdl.linear.basis.PolynomialFunction;
import net.tvburger.jdl.linear.optimizer.*;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.plots.Plot;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class ModelL1Regularization {

    @SuppressWarnings("unchecked")
    public static <O extends LinearModelOptimizer<N> & IterativeOptimizer<N>, N extends Number> void main(String[] args) {
        JavaNumberTypeSupport<N> numberTypeSupport = (JavaNumberTypeSupport<N>) JavaNumberTypeSupport.DOUBLE;
        float lambda1 = 0.001f;
        float lambda2 = 0.0001f;

        VanillaGradientDescentOptimizer<N> gd = new VanillaGradientDescentOptimizer<>(numberTypeSupport);
        VanillaGradientDescentOptimizer<N> l1 = new VanillaGradientDescentOptimizer<>(numberTypeSupport);
        l1.setInterceptor(new L1Regularization<>(numberTypeSupport.valueOf(lambda1)));
        VanillaGradientDescentOptimizer<N> l2 = new VanillaGradientDescentOptimizer<>(numberTypeSupport);
        l2.setInterceptor(new L2Regularization<>(numberTypeSupport.valueOf(lambda2)));
        VanillaGradientDescentOptimizer<N> net = new VanillaGradientDescentOptimizer<>(numberTypeSupport);
        net.setInterceptor(new ElasticNet<>(numberTypeSupport.valueOf(lambda1), numberTypeSupport.valueOf(lambda2)));

//        Map<String, O> optimizers = Map.of("L1(0.01)", (O) l1, "L2(0.01)", (O) l2, "vanilla", (O) gd, "net(0.01,0.01)", (O) net);
        Map<String, O> optimizers = Map.of("L1", (O) l1);
        for (Map.Entry<String, O> optimizerEntry : optimizers.entrySet()) {
            showForNumberType(numberTypeSupport, optimizerEntry.getKey(), optimizerEntry.getValue(), 10_000_000, 0.1f, 3, 0);
        }
    }

    private static <O extends LinearModelOptimizer<N> & IterativeOptimizer<N>, N extends Number> void showForNumberType(JavaNumberTypeSupport<N> typeSupport, String title, O optimizer, int epochs, float learningRate, int m, long sleep) {
        LinearRegression<N> regression = createClosedSolutionFitting(typeSupport, optimizer);

        Plot mrePlot = new Plot("RME " + typeSupport.name() + " [" + title + "] lr=" + learningRate + " m=" + m);
        mrePlot.display();

        Plot fitPlot = new Plot("Fit " + typeSupport.name() + " [" + title + "] lr=" + learningRate + " m=" + m);
        fitPlot.setYRange(-2, 2);
        fitPlot.plotDataSet(regression.getTrainSet(), "Train Set");
        fitPlot.display();

        Plot weightPlot = new Plot("Absolute sum of weights " + typeSupport.name() + " [" + title + "] lr=" + learningRate + " m=" + m);
        weightPlot.display();

        optimizer.setLearningRate(learningRate);
        optimizer.setEpochs(epochs);
        optimizer.setEpochCompletionListener(IterativeOptimizer.initializer(model -> {
//                    model.setParameters((N[]) new Double[]{-0.233, 166.17957, -3589.525, 31330.045, -142827.98, 377871.28, -602523.25, 571137.8, -296352.97, 64788.945});
//                    model.setParameters((N[]) new Double[]{-0.233, 1.17957, -3.525, 3.045, -7.98, 1.28, -6.0, 5.0, -3.0, 6.0});
//                    N factor = model.getCurrentNumberType().valueOf(1.0f);
//                    for (int i = 0; i < model.getParameters().length; i++) {
//                        model.setParameter(i, model.getCurrentNumberType().multiply(model.getParameters()[i], factor));
//                    }
                },
                IterativeOptimizer.sample(250, (epoch, model, trainSet, o) -> {
                    Pair<Float, Map<String, Float>> currentRmes = regression.calculateRMEs(model);
                    mrePlot.addToSeries("Train RME", new float[]{epoch}, new float[]{currentRmes.left()});
                    for (Map.Entry<String, Float> currentRmeEntry : currentRmes.right().entrySet()) {
                        mrePlot.addToSeries(currentRmeEntry.getKey() + " RME", new float[]{epoch}, new float[]{currentRmeEntry.getValue()});
                    }
                    mrePlot.redraw();
                    weightPlot.addToSeries("Total Weights", new float[]{epoch}, new float[]{(float) (Arrays.stream(model.getParameters()).mapToDouble(p -> model.getCurrentNumberType().absolute(p).doubleValue()).sum())});
                    weightPlot.redraw();
                    fitPlot.plotTargetOutput(model, "Fit");
                    fitPlot.redraw();
                    if (epoch == epochs || true) {
                        System.out.println(title + ": " + Arrays.toString(model.getParameters()));
                    }
                    Threads.sleepSilently(sleep);
                })));
        Threads.runAsynchronously(() -> regression.fitComplexity(m));
    }

    private static <N extends Number> LinearRegression<N> createClosedSolutionFitting(JavaNumberTypeSupport<N> typeSupport, LinearModelOptimizer<N> optimizer) {
        SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator = SyntheticDataSets.sinus(typeSupport);
        BasisFunction.Generator<N> basisFunctionGenerator = new PolynomialFunction.Generator<>(typeSupport);
        return create(dataSetGenerator, basisFunctionGenerator, optimizer);
    }

    public static <N extends Number> LinearRegression<N> create(SyntheticDataSets.SyntheticDataSet<N> dataSetGenerator, BasisFunction.Generator<N> basisFunctionGenerator, LinearModelOptimizer<N> optimizer) {
        dataSetGenerator.setRandomSeed(165);
        dataSetGenerator.setNoiseScale(0.5f);
        DataSet<N> trainSet = dataSetGenerator.load(100);
        DataSet<N> testSet = dataSetGenerator.load(10);
        Map<String, DataSet<N>> testSets = new LinkedHashMap<>();
        testSets.put("Test Set", testSet);
        return new LinearRegression<>(basisFunctionGenerator, trainSet, testSets, optimizer);
    }

}
