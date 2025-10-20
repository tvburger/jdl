package net.tvburger.jdl.lenet;

import net.tvburger.jdl.cnn.ConvolutionalLayer;
import net.tvburger.jdl.cnn.ConvolutionalShape;
import net.tvburger.jdl.datasets.MnistDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.scalars.activations.Activations;
import net.tvburger.jdl.plots.ImageViewer;

import java.io.IOException;

public class LeNet {

    public static void main(String[] args) throws IOException {
        DataSet<Float> samples = MnistDataSets.loadDigits();
        for (int i = 0; i < 10; i++) {
            DataSet.Sample<Float> sample = samples.samples().get(i);
            ImageViewer imageViewer = ImageViewer.fromMnistImage(sample);
            imageViewer.display();
            ConvolutionalLayer layer = ConvolutionalLayer.create(28, 28, 1, 5, 1, 0, Activations.tanh());
            ConvolutionalShape input = layer.getInputShape().withPixels(sample.features());
            ConvolutionalShape output = layer.transform(input);
        }
    }

}
