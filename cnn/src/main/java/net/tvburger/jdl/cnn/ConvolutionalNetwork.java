package net.tvburger.jdl.cnn;

import java.util.Arrays;
import java.util.List;

public class ConvolutionalNetwork implements ConvolutionalNetworkLayer {

    private final List<ConvolutionalNetworkLayer> layers;

    public static ConvolutionalNetwork of(ConvolutionalNetworkLayer... layer) {
        return new ConvolutionalNetwork(Arrays.asList(layer));
    }

    public ConvolutionalNetwork(List<ConvolutionalNetworkLayer> layers) {
        this.layers = layers;
    }

    @Override
    public ConvolutionalShape getInputShape() {
        return layers.getFirst().getInputShape();
    }

    @Override
    public ConvolutionalShape getOutputShape() {
        return layers.getLast().getOutputShape();
    }

    @Override
    public ConvolutionalShape transform(ConvolutionalShape input) {
        ConvolutionalShape currentShape = input;
        for (ConvolutionalNetworkLayer layer : layers) {
            currentShape = layer.transform(currentShape);
        }
        return currentShape;
    }

}
