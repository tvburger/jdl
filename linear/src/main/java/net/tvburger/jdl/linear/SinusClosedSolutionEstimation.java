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
import java.util.List;

public class SinusClosedSolutionEstimation {

    public static void main(String[] args) {
        SyntheticDataSets.SyntheticDataSet sinus = SyntheticDataSets.sinus();
        sinus.setRandomSeed(165);
        sinus.setNoiseScale(1.0f);
        DataSet trainSet = sinus.load(10);
        DataSet testSet = sinus.load(1000);

        int from = 0;
        int to = 5;
        float[] rme_train = new float[to - from + 1];
        float[] rme_test_10 = new float[rme_train.length];
        float[] rme_test_100 = new float[rme_train.length];
        float[] rme_test_1000 = new float[rme_train.length];
        float[] sum_of_weights = new float[rme_train.length];
        float[] ms = new float[rme_train.length];
        Plot plot = new Plot("Estimation per N");

        float[] xt = new float[1000];
        float[] yt = new float[xt.length];
        for (int i = 0; i < xt.length; i++) {
            xt[i] = 1.0f * i / 1000;
            yt[i] = sinus.targetOutputs(xt[i]);
        }
        plot.setSeries("target set", xt, yt);

        float[] xtv = new float[1000];
        float[] ytv = new float[xt.length];
        for (int i = 0; i < testSet.size(); i++) {
            xtv[i] = testSet.samples().get(i).features()[0];
            ytv[i] = testSet.samples().get(i).targetOutputs()[0];
        }
        plot.setPoints("test set", xtv, ytv);

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
            Pair<Float, List<Float>> rme = scoreM(m * 2 + 1, trainSet, testSet, plot, table);
            ms[m - from] = (float) m * 2 + 1;
            rme_train[m - from] = rme.left();
            rme_test_10[m - from] = rme.right().get(0);
            rme_test_100[m - from] = rme.right().get(1);
            rme_test_1000[m - from] = rme.right().get(2);
            sum_of_weights[m - from] = rme.right().get(3);
        }
        plot.display();
//        table.display();

        Plot weightPlot = new Plot("Absolute sum of weights");
        weightPlot.setSeries("Weights", ms, sum_of_weights);
        weightPlot.display();

        Plot errorPlot = new Plot("RME per N");
        errorPlot.setSeries("Train RME", ms, rme_train);
        errorPlot.setSeries("Test 10 RME", ms, rme_test_10);
        errorPlot.setSeries("Test 100 RME", ms, rme_test_100);
        errorPlot.setSeries("Test 1000 RME", ms, rme_test_1000);
        errorPlot.display();
    }

    public static Pair<Float, List<Float>> scoreM(int m, DataSet trainSet, DataSet testSet, Plot plot, Table table) {
        PolynomialRegression regression = new PolynomialRegression(m);
        setOptimalWeights(regression, trainSet);
        table.addEntry(new String[]{"" + m, Arrays.toString(regression.getParameters())});

        System.out.println("For m = " + m);
        System.out.println(" id | estim | tgt  | error");
        System.out.println(" --------------------------");
        float mse10 = 0.0f;
        float mse100 = 0.0f;
        float mse1000 = 0.0f;
        int i = 0;
        for (DataSet.Sample sample : testSet) {
            float estimated = regression.estimateScalar(sample.features());
            float target = sample.targetOutputs()[0];
            float mse = (float) Math.pow(estimated - target, 2);
            if (i % 100 == 0) {
                mse10 += mse / 10;
            }
            if (i % 10 == 0) {
                mse100 += mse / 100;
            }
            mse1000 += mse / 1000;
            i++;
//            System.out.printf("%3d | %5.2f | %5.2f | %5.2f\n", i, estimated, target, (estimated - target));
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
        float rme10 = (float) Math.sqrt(mse10);
        float rme100 = (float) Math.sqrt(mse100);
        float rme1000 = (float) Math.sqrt(mse1000);
        float rme_train = 0.0f;
        for (DataSet.Sample train : trainSet) {
            rme_train += (float) Math.pow(regression.estimateUnary(train.features()[0]) - train.targetOutputs()[0], 2) / trainSet.size();
        }
        rme_train = (float) Math.sqrt(rme_train);
        System.out.println("RMS test10 = " + rme10);
        System.out.println("RMS test100 = " + rme100);
        System.out.println("RMS test1000 = " + rme1000);
        System.out.println("RMS train = " + rme_train);
        System.out.println(" --------------------------------------------");
        float total = 0.0f;
        for (float f : regression.getParameters()) {
            total += Math.abs(f);
        }
        return Pair.of(rme_train, List.of(rme10, rme100, rme1000, total));
    }

    private static void setOptimalWeights(PolynomialRegression regression, DataSet trainSet) {
        Matrix<Float> designMatrix = FeatureMatrices.create(regression.getFeatureExtractor(), trainSet);
        Matrix<Double> designMatrixDouble = Matrices.withDoublePrecision(designMatrix);
        designMatrix.print("Φ");
        float[] values = new float[trainSet.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = trainSet.samples().get(i).targetOutputs()[0];
        }
        Vector<Float> y = Vectors.of(values).transpose();
        y.print("y");
        Matrix<Double> invertedDesignMatrixDouble = designMatrixDouble.pseudoInvert();
        invertedDesignMatrixDouble.multiply(designMatrixDouble).print("double precision: Φ⁻¹Φ = I");
        Vector<Float> weights = Matrices.withSinglePrecision(invertedDesignMatrixDouble).multiply(y);
        weights.print("w");
        System.out.println(Arrays.toString(weights.asArray()));
        for (int i = 0; i < weights.getDimensions(); i++) {
            regression.setParameter(i, weights.get(i + 1));
        }
    }

}
