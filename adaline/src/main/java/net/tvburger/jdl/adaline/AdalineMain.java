package net.tvburger.jdl.adaline;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.datasets.LinesAndCircles;
import net.tvburger.jdl.datasets.LogicalDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regimes.ChainedRegime;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.plots.ImageViewer;
import net.tvburger.jdl.plots.listeners.EpochRmePlotter;

import java.util.Arrays;

public class AdalineMain {

    public static void main(String[] args) {
        DataSet<Float> dataSet = LogicalDataSets.toMinusSet(new LinesAndCircles().load());
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);

        Adaline adaline = Adaline.create(400, 8);
        ObjectiveFunction<Float> objective = Objectives.mSE(JavaNumberTypeSupport.FLOAT);
        LeastMeanSquares leastMeanSquares = new LeastMeanSquares(0.01f);
        EpochRmePlotter epochRmePlotter = new EpochRmePlotter();
        epochRmePlotter.display();
        ChainedRegime regime = Regimes.epochs(100,
                epochRmePlotter.attach("Training Set (" + trainingSet.size() + ")"),
                epochRmePlotter.attach("Test Set (" + testSet.size() + ")", testSet)).stochastic();
        Trainer<Adaline, Float> adalineTrainer = Trainer.of(new AdalineInitializer(), objective, leastMeanSquares, regime);
        adalineTrainer.train(adaline, trainingSet);

        int i = 0;
        int correct = 0;
        int wrong = 0;
        for (DataSet.Sample<Float> sample : testSet) {
            i++;
            boolean[] estimate = adaline.classify(sample.features());
            boolean[] target = Floats.toBooleans(sample.targetOutputs());
            if (!Arrays.equals(target, estimate)) {
                wrong++;
                String label = createLabel(i, target, estimate);
                ImageViewer image = ImageViewer.fromPerceptronImage(label, fromMinusSet(sample));
                image.display();
            } else {
                correct++;
            }
        }
        if (wrong == 0) {
            DataSet.Sample<Float> sample = testSet.samples().getFirst();
            boolean[] estimate = adaline.classify(sample.features());
            boolean[] target = Floats.toBooleans(sample.targetOutputs());
            String label = createLabel(1, target, estimate);
            ImageViewer image = ImageViewer.fromPerceptronImage(label, fromMinusSet(sample));
            image.display();
        }
        System.out.println("Correct: " + correct + ", Wrong: " + wrong + ", Total: " + (correct + wrong));
    }

    private static DataSet.Sample<Float> fromMinusSet(DataSet.Sample<Float> sample) {
        Float[] features = sample.features();
        for (int i = 0; i < features.length; i++) {
            if (Floats.equals(features[i], -1.0f)) {
                features[i] = 0.0f;
            }
        }
        return new DataSet.Sample<>(features, sample.targetOutputs());
    }

    private static String createLabel(int i, boolean[] target, boolean[] estimate) {
        String label = "";
        boolean wrong = false;
        label += estimate[0] ? "Circle " : "Line ";
        if (target[0] != estimate[0]) {
            wrong = true;
        }
        label += estimate[1] ? "Left" : "Right";
        if (target[1] != estimate[1]) {
            wrong = true;
        }
        return i + ". " + (wrong ? "X: " : "V: ") + label;
    }

}
