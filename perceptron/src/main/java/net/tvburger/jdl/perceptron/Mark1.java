package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.LinesAndCircles;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.plots.ImageViewer;
import net.tvburger.jdl.plots.listeners.EpochRmePlotter;

public class Mark1 {

    public static void main(String[] args) {
        Perceptron mark1 = Perceptron.create(400, 512, 8);
        mark1.accept(new PerceptronInitializer());

        DataSet<Float> dataSet = new LinesAndCircles().load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);
        EpochRmePlotter epochRmePlotter = new EpochRmePlotter();
        epochRmePlotter.display();
        Regime regime = Regimes.epochs(10,
                epochRmePlotter.attach("Training Set (" + trainingSet.size() + ")"),
                epochRmePlotter.attach("Test Set (" + testSet.size() + ")", testSet)).stochastic();
        Trainer<Perceptron, Float> trainer = Trainer.of(new PerceptronInitializer(), null, new PerceptronUpdateRule(), regime);
        trainer.train(mark1, trainingSet);

        int i = 0;
        int correct = 0;
        int wrong = 0;
        for (DataSet.Sample<Float> sample : testSet) {
            i++;
            Array<Float> estimate = mark1.estimate(sample.features());
            if (Array.equals(estimate, sample.targetOutputs())) {
                correct++;
            } else {
                String label = createLabel(i, sample, estimate);
                ImageViewer image = ImageViewer.fromPerceptronImage(label, sample);
                image.display();
                wrong++;
            }
        }
        if (wrong == 0) {
            DataSet.Sample<Float> sample = testSet.samples().getFirst();
            Array<Float> estimate = mark1.estimate(sample.features());
            String label = createLabel(1, sample, estimate);
            ImageViewer image = ImageViewer.fromPerceptronImage(label, sample);
            image.display();
        }
        System.out.println("Correct: " + correct + ", Wrong: " + wrong + ", Total: " + (correct + wrong));
    }

    private static String createLabel(int i, DataSet.Sample<Float> sample, Array<Float> estimate) {
        String label = "";
        boolean wrong = false;
        label += estimate.get(0) == 1.0f ? "Circle " : "Line ";
        if (!JavaNumberTypeSupport.FLOAT.equals(sample.targetOutputs().get(0), estimate.get(0))) {
            wrong = true;
        }
        label += estimate.get(1) == 1.0f ? "Left" : "Right";
        if (!JavaNumberTypeSupport.FLOAT.equals(sample.targetOutputs().get(1), estimate.get(1))) {
            wrong = true;
        }
        return i + ". " + (wrong ? "X: " : "V: ") + label;
    }

}
