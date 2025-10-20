package net.tvburger.jdl.mlp;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.datasets.LinesAndCircles;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.nn.training.initializers.XavierInitializer;
import net.tvburger.jdl.model.nn.training.optimizers.NeuralNetworkOptimizers;
import net.tvburger.jdl.model.scalars.activations.Activations;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.Trainer;
import net.tvburger.jdl.model.training.loss.Objectives;
import net.tvburger.jdl.model.training.regimes.Regimes;
import net.tvburger.jdl.plots.ImageViewer;
import net.tvburger.jdl.plots.listeners.EpochRmePlotter;

import java.util.Arrays;

public class MLPMark1Main {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), Activations.linear(), 400, 512, 8);

        DataSet<Float> dataSet = new LinesAndCircles().load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);
        EpochRmePlotter epochRmePlotter = new EpochRmePlotter();
        epochRmePlotter.display();
        Regime regime = Regimes.epochs(2,
                epochRmePlotter.attach("Training Set (" + trainingSet.size() + ")"),
                epochRmePlotter.attach("Test Set (" + testSet.size() + ")", testSet)).stochastic();
        Optimizer<? super MultiLayerPerceptron, Float> optimizer = NeuralNetworkOptimizers.vanilla();
        Trainer<MultiLayerPerceptron, Float> trainer = Trainer.of(new XavierInitializer(), Objectives.bCE(mlp.getCurrentNumberType()), optimizer, regime);
        trainer.train(mlp, trainingSet);

        int i = 0;
        int correct = 0;
        int wrong = 0;
        for (DataSet.Sample<Float> sample : testSet) {
            i++;
            Float[] estimate = mlp.estimate(sample.features());
            if (Arrays.equals(estimate, sample.targetOutputs())) {
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
            Float[] estimate = mlp.estimate(sample.features());
            String label = createLabel(1, sample, estimate);
            ImageViewer image = ImageViewer.fromPerceptronImage(label, sample);
            image.display();
        }
        System.out.println("Correct: " + correct + ", Wrong: " + wrong + ", Total: " + (correct + wrong));
    }

    private static String createLabel(int i, DataSet.Sample<Float> sample, Float[] estimate) {
        String label = "";
        boolean wrong = false;
        label += estimate[0] == 1.0f ? "Circle " : "Line ";
        if (!JavaNumberTypeSupport.FLOAT.equals(sample.targetOutputs()[0], estimate[0])) {
            wrong = true;
        }
        label += estimate[1] == 1.0f ? "Left" : "Right";
        if (!JavaNumberTypeSupport.FLOAT.equals(sample.targetOutputs()[1], estimate[1])) {
            wrong = true;
        }
        return i + ". " + (wrong ? "X: " : "V: ") + label;
    }

}
