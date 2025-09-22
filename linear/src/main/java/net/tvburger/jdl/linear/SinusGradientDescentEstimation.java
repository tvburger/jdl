package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.datasets.SyntheticDataSets;
import net.tvburger.jdl.linalg.Matrices;
import net.tvburger.jdl.linalg.Matrix;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linalg.Vectors;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.plots.Plot;
import net.tvburger.jdl.plots.Table;

import java.util.Arrays;

public class SinusGradientDescentEstimation {

    public static int EPOCHS = 1000;
    public static double LEARNING_RATE = 0.1f;

    public static void main(String[] args) {
//        SyntheticDataSets.SyntheticDataSet<Float> sinus = SyntheticDataSets.sinus(JavaNumberTypeSupport.FLOAT);
//        sinus.setRandomSeed(165);
//        sinus.setNoiseScale(1.0f);
//        DataSet<Float> trainSet = sinus.load(5);
//        DataSet<Float> testSet = sinus.load(10);
//
//        int from = 4;
//        int to = 4;
//        Float[] rme_train = new float[to - from + 1];
//        Float[] rme_test = new float[rme_train.length];
//        Float[] ms = new float[rme_train.length];
//        Plot plot = new Plot("Estimation per N");
//
//        Float[] xt = new Float[1000];
//        Float[] yt = new Float[xt.length];
//        for (int i = 0; i < xt.length; i++) {
//            xt[i] = 1.0f * i / 1000;
//            yt[i] = sinus.getEstimationFunction().estimate(xt[i]);
//        }
//        plot.setSeries("target set", xt, yt);
//
//        float[] x = new float[trainSet.size()];
//        float[] r = new float[trainSet.size()];
//        for (int n = 0; n < x.length; n++) {
//            x[n] = trainSet.samples().get(n).features()[0];
//            r[n] = trainSet.samples().get(n).targetOutputs()[0];
//        }
//        plot.setPoints("train", x, r);
//
//        Table table = new Table("Weights");
//        table.setColumnNames("M", "Weights");
//        for (int m = from; m <= to; m += 1) {
//            Pair<Float, Float> rme = scoreM(m, trainSet, testSet, plot, table);
//            ms[m - from] = (float) m;
//            rme_train[m - from] = rme.left();
//            rme_test[m - from] = rme.right();
//        }
//        plot.display();
//        table.display();
//
//        Plot errorPlot = new Plot("RME per N");
//        errorPlot.setSeries("Train RME", ms, rme_train);
//        errorPlot.setSeries("Test RME", ms, rme_test);
//        errorPlot.display();
    }
//
//    public static Pair<Float, Float> scoreM(int m, DataSet<Float> trainSet, DataSet<Float> testSet, Plot plot, Table table) {
//        PolynomialRegression<Float> regression = new PolynomialRegression<>(m, JavaNumberTypeSupport.FLOAT);
//        setOptimalWeights(regression, trainSet);
//        table.addEntry(new String[]{"" + m, Arrays.toString(regression.getParameters())});
//
//        System.out.println("For m = " + m);
//        System.out.println(" id | estim | tgt  | error");
//        System.out.println(" --------------------------");
//        float mse = 0.0f;
//        int i = 0;
//        for (DataSet.Sample<Float> sample : testSet) {
//            float estimated = regression.estimateScalar(sample.features());
//            float target = sample.targetOutputs()[0];
//            mse += (float) Math.pow(estimated - target, 2) / testSet.size();
//            System.out.printf("%3d | %5.2f | %5.2f | %5.2f\n", ++i, estimated, target, (estimated - target));
//        }
//        float[] x = new float[1000];
//        float[] y = new float[1000];
//        for (int n = 0; n < x.length; n++) {
//            x[n] = ((float) n) / (x.length - 1);
//            y[n] = regression.estimateUnary(x[n]);
//        }
//        if (m % 2 != 0 || m == 0) {
//            plot.setSeries("m = " + m, x, y);
//        }
//        System.out.println("Parameters: " + Arrays.toString(regression.getParameters()));
//        float rme = (float) Math.sqrt(mse);
//        float rme_train = 0.0f;
//        for (DataSet.Sample<Float> train : trainSet) {
//            rme_train += (float) Math.pow(regression.estimateUnary(train.features()[0]) - train.targetOutputs()[0], 2) / trainSet.size();
//        }
//        rme_train = (float) Math.sqrt(rme_train);
//        System.out.println("RMS test = " + rme);
//        System.out.println("RMS train = " + rme_train);
//        System.out.println(" --------------------------------------------");
//        return Pair.of(rme_train, rme);
//    }

    private static void setOptimalWeights(PolynomialRegression<Float> regression, DataSet<Float> trainSet) {
        Matrix<Double> designMatrix = Matrices.withDoublePrecision(FeatureMatrices.create(regression.getFeatureExtractor(), trainSet));
        float[] values = new float[trainSet.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = trainSet.samples().get(i).targetOutputs()[0];
        }
        Vector<Double> y = Vectors.withDoublePrecision(Vectors.of(values).transpose());
        for (int e = 0; e < EPOCHS; e++) {
            float[] parameters = new float[regression.getParameterCount()];
            for (int i = 0; i < parameters.length; i++) {
                parameters[i] = regression.getParameter(i);
            }
            Vector<Double> thetas = Vectors.withDoublePrecision(Vectors.of(parameters).transpose());
            Vector<Double> error = designMatrix.multiply(thetas).substract(y);
            Vector<Double> adjustment = designMatrix.transpose().multiply(error).multiply(1.0 / trainSet.size()).multiply(LEARNING_RATE);
            thetas = thetas.substract(adjustment);
            for (int i = 0; i < parameters.length; i++) {
                regression.setParameter(i, (float) (double) thetas.get(i + 1));
            }
        }
    }

}
