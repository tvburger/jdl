package net.tvburger.jdl.cnn;

import java.util.ArrayList;
import java.util.List;

// For now only supports 2D pooling
public class PoolingLayer implements ConvolutionalNetworkLayer {

    private final ConvolutionalShape inputShape;
    private final ConvolutionalShape outputShape;
    private final PoolingFunction poolingFunction;
    private final int size;
    private final int stride;
    private final int padding;

    public static PoolingLayer create(int width, int height, int size, PoolingFunction poolingFunction) {
        ConvolutionalShape inputShape = new ConvolutionalShape(width, height, ConvolutionalShape.DEFAULT_CHANNELS, ConvolutionalShape.DEFAULT_COUNT, null);
        return create(inputShape, size, poolingFunction);
    }

    public static PoolingLayer create(ConvolutionalShape inputShape, int size, PoolingFunction poolingFunction) {
        return create(inputShape, size, size, 0, poolingFunction);
    }

    public static PoolingLayer create(ConvolutionalShape inputShape, int size, int stride, int padding, PoolingFunction poolingFunction) {
        int outputWidth = (inputShape.getWidth() - size + 2 * padding) / stride + 1;
        int outputHeight = (inputShape.getHeight() - size + 2 * padding) / stride + 1;
        ConvolutionalShape outputShape = new ConvolutionalShape(outputWidth, outputHeight, ConvolutionalShape.DEFAULT_CHANNELS, ConvolutionalShape.DEFAULT_COUNT, null);
        return new PoolingLayer(inputShape, outputShape, poolingFunction, size, stride, padding);
    }

    private PoolingLayer(ConvolutionalShape inputShape, ConvolutionalShape outputShape, PoolingFunction poolingFunction, int size, int stride, int padding) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.poolingFunction = poolingFunction;
        this.size = size;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public ConvolutionalShape getInputShape() {
        return inputShape;
    }

    @Override
    public ConvolutionalShape getOutputShape() {
        return outputShape;
    }

    public PoolingFunction getPoolingFunction() {
        return poolingFunction;
    }

    public int getSize() {
        return size;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    @Override
    public ConvolutionalShape transform(ConvolutionalShape input) {
        ConvolutionalShape output = getOutputShape().clone();
        for (int i = 0; i < output.getCount(); i++) {
            for (int c = 1; c <= output.getChannels(); c++) {
                for (int y = 0; y < output.getHeight(); y++) {
                    for (int x = 0; x < output.getWidth(); x++) {
                        for (ConvolutionalShape window : getWindows(input)) {
                            float value = poolingFunction.pool(window.getElements());
                            output.setElement(value, x, y, c, i);
                        }
                    }
                }
            }
        }
        return output;
    }

    private List<ConvolutionalShape> getWindows(ConvolutionalShape input) {
        List<ConvolutionalShape> windows = new ArrayList<>();
        for (int x = -padding; x <= input.getWidth() - size + padding; x = x + stride) {
            for (int y = -padding; y <= input.getHeight() - size + padding; y = y + stride) {
                for (int c = 1; c <= input.getChannels(); c++) {
                    for (int i = 0; i < input.getCount(); i++) {
                        ConvolutionalShape window = new ConvolutionalShape(size, size);
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
