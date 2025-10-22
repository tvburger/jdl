package net.tvburger.jdl.cnn;

import net.tvburger.jdl.model.scalars.activations.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalLayer implements ConvolutionalNetworkLayer {

    private final ConvolutionalShape inputShape;
    private final ConvolutionalShape outputShape;
    private final List<ConvolutionalFilter> filters;
    private final int stride;
    private final int padding;
    private final ActivationFunction activationFunction;

    public static ConvolutionalLayer create(int width, int height, ActivationFunction activationFunction) {
        return create(width, height, 1, activationFunction);
    }

    public static ConvolutionalLayer create(int width, int height, int depth, ActivationFunction activationFunction) {
        return create(width, height, depth, 3, 1, 0, activationFunction);
    }

    public static ConvolutionalLayer create(int width, int height, int depth, int size, int stride, int padding, ActivationFunction activationFunction) {
        ConvolutionalShape inputShape = new ConvolutionalShape(width, height, depth, 1, null);
        return create(inputShape, size, stride, 1, padding, activationFunction);
    }

    public static ConvolutionalLayer create(ConvolutionalShape inputShape, int size, int filters, int stride, int padding, ActivationFunction activationFunction) {
        int outputWidth = (inputShape.getWidth() - size + 2 * padding) / stride + 1;
        int outputHeight = (inputShape.getHeight() - size + 2 * padding) / stride + 1;
        ConvolutionalShape outputShape = new ConvolutionalShape(outputWidth, outputHeight, filters, 1, null);
        return create(inputShape, outputShape, filters, stride, padding, activationFunction);
    }

    public static ConvolutionalLayer create(ConvolutionalShape inputShape, ConvolutionalShape outputShape, int size, int stride, int padding, ActivationFunction activationFunction) {
        List<ConvolutionalFilter> filters = new ArrayList<>();
        for (int i = 0; i < outputShape.getChannels(); i++) {
            filters.add(ConvolutionalFilter.create(size, inputShape.getChannels()));
        }
        return new ConvolutionalLayer(inputShape, outputShape, filters, stride, padding, activationFunction);
    }

    private ConvolutionalLayer(ConvolutionalShape inputShape, ConvolutionalShape outputShape, List<ConvolutionalFilter> filters, int stride, int padding, ActivationFunction activationFunction) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.filters = filters;
        this.stride = stride;
        this.padding = padding;
        this.activationFunction = activationFunction;
    }

    @Override
    public ConvolutionalShape getInputShape() {
        return inputShape;
    }

    @Override
    public ConvolutionalShape getOutputShape() {
        return outputShape;
    }

    public List<ConvolutionalFilter> getFilters() {
        return filters;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    public int getDimensions() {
        return filters.size();
    }

    @Override
    public ConvolutionalShape transform(ConvolutionalShape input) {
        ConvolutionalShape output = getOutputShape().clone();
        for (int d = 0; d < output.getChannels(); d++) {
            ConvolutionalFilter filter = filters.get(d);
            for (int y = 0; y < output.getHeight(); y++) {
                for (int x = 0; x < output.getWidth(); x++) {
                    for (ConvolutionalShape window : getWindows(input)) {
                        float value = filter.applyToWindow(window);
                        if (activationFunction != null) {
                            value = activationFunction.activate(value);
                        }
                        output.setElement(value, x, y, d + 1);
                    }
                }
            }
        }
        return output;
    }

    private List<ConvolutionalShape> getWindows(ConvolutionalShape input) {
        if (filters.isEmpty()) {
            return List.of();
        }
        ConvolutionalShape filterSize = filters.getFirst().getShape();
        List<ConvolutionalShape> windows = new ArrayList<>();
        for (int x = -padding; x <= input.getWidth() - filterSize.getWidth() + padding; x = x + stride) {
            for (int y = -padding; y <= input.getHeight() - filterSize.getHeight() + padding; y = y + stride) {
                for (int c = 1; c <= input.getChannels(); c++) {
                    for (int i = 0; i < input.getCount(); i++) {
                        ConvolutionalShape window = filterSize.clone();
                        for (int yi = 0; yi < window.getHeight(); yi++) {
                            for (int xi = 0; xi < window.getWidth(); xi++) {
                                float value;
                                if (x + xi < 0 || x + xi >= input.getWidth() || y + yi < 0 || y + yi >= input.getHeight()) {
                                    value = 0f;
                                } else {
                                    value = input.getElement(x + xi, y + yi, c, i);
                                }
                                window.setElement(value, xi, yi, c, i);
                            }
                        }
                        windows.add(window);
                    }
                }
            }
        }
        return windows;
    }
}
