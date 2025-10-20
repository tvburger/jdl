package net.tvburger.jdl.mlp;

import net.tvburger.jdl.common.utils.Floats;
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

public class MLPCirclesAndSquaresMain {

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = MultiLayerPerceptron.create(Activations.sigmoid(), Activations.sigmoid(), 400, 8);

        DataSet<Float> dataSet = new LinesAndCircles().load();
        DataSet<Float> trainingSet = dataSet.subset(10, dataSet.size());
        DataSet<Float> testSet = dataSet.subset(0, 10);
        EpochRmePlotter epochRmePlotter = new EpochRmePlotter();
        epochRmePlotter.display();
        Regime regime = Regimes.epochs(100,
                epochRmePlotter.attach("Training Set (" + trainingSet.size() + ")"),
                epochRmePlotter.attach("Test Set (" + testSet.size() + ")", testSet)).batch();
        Optimizer<? super MultiLayerPerceptron, Float> optimizer = NeuralNetworkOptimizers.adaGrad(0.1f);
        Trainer<MultiLayerPerceptron, Float> trainer = Trainer.of(new XavierInitializer(), Objectives.bCE(mlp.getCurrentNumberType()), optimizer, regime);
        trainer.train(mlp, trainingSet);

        int i = 0;
        int correct = 0;
        int wrong = 0;
        for (DataSet.Sample<Float> sample : testSet) {
            i++;
            boolean[] estimate = Floats.toBooleans(mlp.estimate(sample.features()), 0.5f);
            boolean[] target = Floats.toBooleans(sample.targetOutputs(), 0.5f);
            if (Arrays.equals(estimate, target)) {
                correct++;
            } else {
                String label = createLabel(i, target, estimate);
                ImageViewer image = ImageViewer.fromPerceptronImage(label, sample);
                image.display();
                wrong++;
            }
        }
        if (wrong == 0) {
            DataSet.Sample<Float> sample = testSet.samples().getFirst();
            boolean[] estimate = Floats.toBooleans(mlp.estimate(sample.features()), 0.5f);
            boolean[] target = Floats.toBooleans(sample.targetOutputs(), 0.5f);
            String label = createLabel(1, target, estimate);
            ImageViewer image = ImageViewer.fromPerceptronImage(label, sample);
            image.display();
        }
        System.out.println("Correct: " + correct + ", Wrong: " + wrong + ", Total: " + (correct + wrong));
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
