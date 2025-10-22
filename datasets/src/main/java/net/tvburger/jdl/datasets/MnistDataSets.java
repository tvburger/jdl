package net.tvburger.jdl.datasets;

import net.tvburger.jdl.cnn.ConvolutionalShape;
import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.model.DataSet;

import java.io.IOException;
import java.util.Arrays;

public final class MnistDataSets {

    private MnistDataSets() {
    }

    public static ConvolutionalShape digitAsShape(Array<Float> features) throws IOException {
        return new ConvolutionalShape(28, 28, ConvolutionalShape.DEFAULT_CHANNELS, ConvolutionalShape.DEFAULT_INDEX, features);
    }

    public static DataSet<Float> loadDigits() throws IOException {
        MnistReader.MnistData mnistData = MnistReader.readImagesLabels(
                "mnist/train-images.idx3-ubyte",
                "mnist/train-labels.idx1-ubyte");
        DataSet.Sample<Float>[] samples = new DataSet.Sample[mnistData.images.size()];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = DataSet.Sample.of(
                    imageToFloatFeatures(mnistData.images.get(i)),
                    labelToFloatOutputs(mnistData.labels[i]));
        }
        return DataSet.of(samples);
    }

    private static Array<Float> imageToFloatFeatures(int[] imagePixels) {
        Float[] features = new Float[imagePixels.length];
        for (int i = 0; i < imagePixels.length; i++) {
            features[i] = imagePixels[i] / 255.0f;
        }
        return Array.of(features);
    }

    private static Array<Float> labelToFloatOutputs(byte label) {
        Float[] outputs = new Float[10];
        Arrays.fill(outputs, 0.0f);
        outputs[label] = 1.0f;
        return Array.of(outputs);
    }

}
