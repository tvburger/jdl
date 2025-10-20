package net.tvburger.jdl.cnn;

import net.tvburger.jdl.model.scalars.activations.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalLayer {

    private final ConvolutionalShape inputShape;
    private final ConvolutionalShape outputShape;
    private final List<ConvolutionalFilter> filters;
    private final int stride;
    private final int padding;

    public static ConvolutionalLayer create(int width, int height, ActivationFunction activationFunction) {
        return create(width, height, 1, activationFunction);
    }

    public static ConvolutionalLayer create(int width, int height, int depth, ActivationFunction activationFunction) {
        return create(width, height, depth, 3, 1, 0, activationFunction);
    }

    public static ConvolutionalLayer create(int width, int height, int depth, int size, int stride, int padding, ActivationFunction activationFunction) {
        ConvolutionalShape inputShape = new ConvolutionalShape(width, height, depth, 1, null);
        return create(inputShape, size, stride, padding, 1, activationFunction);
    }

    public static ConvolutionalLayer create(ConvolutionalShape inputShape, int size, int dimensions, int stride, int padding, ActivationFunction activationFunction) {
        int outputWidth = (inputShape.getWidth() - size + 2 * padding) / stride + 1;
        int outputHeight = (inputShape.getHeight() - size + 2 * padding) / stride + 1;
        ConvolutionalShape outputShape = new ConvolutionalShape(outputWidth, outputHeight, dimensions, 1, null);
        return create(inputShape, outputShape, dimensions, stride, padding, activationFunction);
    }

    public static ConvolutionalLayer create(ConvolutionalShape inputShape, ConvolutionalShape outputShape, int size, int stride, int padding, ActivationFunction activationFunction) {
        List<ConvolutionalFilter> filters = new ArrayList<>();
        for (int i = 0; i < outputShape.getChannels(); i++) {
            filters.add(ConvolutionalFilter.create(size, inputShape.getChannels()));
        }
        return new ConvolutionalLayer(inputShape, outputShape, filters, stride, padding);
    }

    private ConvolutionalLayer(ConvolutionalShape inputShape, ConvolutionalShape outputShape, List<ConvolutionalFilter> filters, int stride, int padding) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.filters = filters;
        this.stride = stride;
        this.padding = padding;
    }

    public ConvolutionalShape getInputShape() {
        return inputShape;
    }

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

    public ConvolutionalShape transform(ConvolutionalShape input) {
        ConvolutionalShape output = getOutputShape().clone();
        for (int d = 0; d < getDimensions(); d++) {
            ConvolutionalFilter filter = filters.get(d);
            for (int y = 0; y < output.getHeight(); y++) {
                for (int x = 0; x < output.getWidth(); x++) {
                    for (Float[] field : getFields(input, output)) {
                        float value = filter.estimate(field)[0];
                        output.setPixel(value, x, y, d + 1);
                    }
                }
            }
        }
        return output;
    }

    private List<Float[]> getFields(ConvolutionalShape input, ConvolutionalShape output) {
        if (filters.isEmpty()) {
            return List.of();
        }
        ConvolutionalShape filterSize = filters.getFirst().getShape();
        List<Float[]> fields = new ArrayList<>();
        for (int x = -padding; x < output.getWidth() - filterSize.getWidth() + padding; x = x + stride) {
            for (int y = -padding; y < input.getWidth() - filterSize.getHeight() + padding; y = y + stride) {
                for (int c = 1; c <= input.getChannels(); c++) {
                    for (int i = 0; i < input.getCount(); i++) {
                        ConvolutionalShape fieldShape = filterSize.clone();
                        for (int yi = 0; yi < fieldShape.getHeight(); yi++) {
                            for (int xi = 0; xi < fieldShape.getWidth(); xi++) {
                                float value;
                                if (x + xi < 0 || x + xi >= input.getWidth() || y + yi < 0 || y + yi >= input.getHeight()) {
                                    value = 0f;
                                } else {
                                    value = input.getPixel(x + xi, y + yi, c, i);
                                }
                                fieldShape.setPixel(value, xi, yi, c, i);
                            }
                        }
                        fields.add(fieldShape.getPixels());
                    }
                }
            }
        }
        return fields;
    }
}
