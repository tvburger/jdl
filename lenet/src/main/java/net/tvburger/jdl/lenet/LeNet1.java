package net.tvburger.jdl.lenet;

import net.tvburger.jdl.cnn.*;
import net.tvburger.jdl.cnn.pooling.AveragePooling;
import net.tvburger.jdl.datasets.MnistDataSets;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.scalars.activations.Activations;
import net.tvburger.jdl.plots.ImageViewer;

import java.io.IOException;

public class LeNet1 {

    public static void main(String[] args) throws IOException {
        ConvolutionalLayer layer1 = ConvolutionalLayer.create(new ConvolutionalShape(16, 16), 5, 6, 1, 0, Activations.tanh());
        PoolingLayer layer2 = PoolingLayer.create(layer1.getOutputShape(), 2, new AveragePooling());
        ConvolutionalLayer layer3 = ConvolutionalLayer.create(layer2.getOutputShape(), 5, 12, 1, 0, Activations.tanh());
        FullyConnectedLayer layer4 = FullyConnectedLayer.create(layer3.getOutputShape(), 10, Activations.none());
        ConvolutionalNetwork convolutionalNetwork = ConvolutionalNetwork.of(layer1, layer2, layer3, layer4);

        DataSet<Float> samples = MnistDataSets.loadDigits();
        for (int i = 0; i < 10; i++) {
            DataSet.Sample<Float> sample = samples.samples().get(i);
            ImageViewer imageViewer = ImageViewer.fromMnistImage(sample);
            imageViewer.display();

//            ConvolutionalShape input = MnistDataSets.digitAsShape(sample.features());
//            ConvolutionalShape output = convolutionalNetwork.transform(input);
//            int digit = shapeAsDigit(output);
        }
    }

    private static int shapeAsDigit(ConvolutionalShape output) {
        Float max = null;
        int digit = -1;
        for (int i = 0; i < output.getElements().length(); i++) {
            if (max == null || output.getElements().get(i) > max) {
                max = output.getElements().get(i);
                digit = i;
            }
        }
        return digit;
    }
}
