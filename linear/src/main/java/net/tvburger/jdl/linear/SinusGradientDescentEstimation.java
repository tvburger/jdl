package net.tvburger.jdl.linear;

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
    public static double LEARNING_RATE = 0.5f;

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet sinus = SyntheticDataSets.sinus();
        sinus.setRandomSeed(10);
        sinus.setNoiseScale(0.5f);
        DataSet trainSet = sinus.load(10);
        DataSet testSet = sinus.load(10);

        int from = 0;
        int to = 9;
        float[] rme_train = new float[to - from + 1];
        float[] rme_test = new float[rme_train.length];
        float[] ms = new float[rme_train.length];
        Plot plot = new Plot("Estimation per N");

        float[] xt = new float[1000];
        float[] yt = new float[xt.length];
        for (int i = 0; i < xt.length; i++) {
            xt[i] = 1.0f * i / 1000;
            yt[i] = sinus.targetOutputs(xt[i]);
        }
        plot.setSeries("target set", xt, yt);

        float[] x = new float[trainSet.size()];
        float[] r = new float[trainSet.size()];
        for (int n = 0; n < x.length; n++) {
            x[n] = trainSet.samples().get(n).features()[0];
            r[n] = trainSet.samples().get(n).targetOutputs()[0];
        }
        plot.setPoints("train", x, r);

        Table table = new Table("Weights");
        table.setColumnNames("M", "Weights");
        for (int m = from; m <= to; m += 1) {
            Pair<Float, Float> rme = scoreM(m, trainSet, testSet, plot, table);
            ms[m - from] = (float) m;
            rme_train[m - from] = rme.left();
            rme_test[m - from] = rme.right();
        }
        plot.display();
        table.display();

        Plot errorPlot = new Plot("RME per N");
        errorPlot.setSeries("Train RME", ms, rme_train);
        errorPlot.setSeries("Test RME", ms, rme_test);
        errorPlot.display();
    }

    public static Pair<Float, Float> scoreM(int m, DataSet trainSet, DataSet testSet, Plot plot, Table table) {
        PolynomialRegression regression = new PolynomialRegression(m);
        setOptimalWeights(regression, trainSet);
        table.addEntry(new String[]{"" + m, Arrays.toString(regression.getParameters())});

        System.out.println("For m = " + m);
        System.out.println(" id | estim | tgt  | error");
        System.out.println(" --------------------------");
        float mse = 0.0f;
        int i = 0;
        for (DataSet.Sample sample : testSet) {
            float estimated = regression.estimateScalar(sample.features());
            float target = sample.targetOutputs()[0];
            mse += (float) Math.pow(estimated - target, 2) / testSet.size();
            System.out.printf("%3d | %5.2f | %5.2f | %5.2f\n", ++i, estimated, target, (estimated - target));
        }
        float[] x = new float[1000];
        float[] y = new float[1000];
        for (int n = 0; n < x.length; n++) {
            x[n] = ((float) n) / (x.length - 1);
            y[n] = regression.estimateUnary(x[n]);
        }
        if (m % 2 != 0 || m == 0) {
            plot.setSeries("m = " + m, x, y);
        }
        System.out.println("Parameters: " + Arrays.toString(regression.getParameters()));
        float rme = (float) Math.sqrt(mse);
        float rme_train = 0.0f;
        for (DataSet.Sample train : trainSet) {
            rme_train += (float) Math.pow(regression.estimateUnary(train.features()[0]) - train.targetOutputs()[0], 2) / trainSet.size();
        }
        rme_train = (float) Math.sqrt(rme_train);
        System.out.println("RMS test = " + rme);
        System.out.println("RMS train = " + rme_train);
        System.out.println(" --------------------------------------------");
        return Pair.of(rme_train, rme);
    }

    private static void setOptimalWeights(PolynomialRegression regression, DataSet trainSet) {
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
            thetas = thetas.substract(designMatrix.transpose().multiply(designMatrix.multiply(thetas).substract(y)).multiply(LEARNING_RATE / trainSet.size()));
            for (int i = 0; i < parameters.length; i++) {
                regression.setParameter(i, (float) (double) thetas.get(i + 1));
            }
        }
    }

}
