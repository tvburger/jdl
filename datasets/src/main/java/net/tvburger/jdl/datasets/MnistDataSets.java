package net.tvburger.jdl.datasets;

import net.tvburger.jdl.model.DataSet;

import java.io.IOException;
import java.util.Arrays;

public final class MnistDataSets {

    private MnistDataSets() {
    }

    @SuppressWarnings("unchecked")
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

    private static Float[] imageToFloatFeatures(int[] imagePixels) {
        Float[] features = new Float[imagePixels.length];
        for (int i = 0; i < imagePixels.length; i++) {
            features[i] = imagePixels[i] / 255.0f;
        }
        return features;
    }

    private static Float[] labelToFloatOutputs(byte label) {
        Float[] outputs = new Float[10];
        Arrays.fill(outputs, 0.0f);
        outputs[label] = 1.0f;
        return outputs;
    }

}
