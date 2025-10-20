package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.shapes.Shape2D;
import net.tvburger.jdl.model.scalars.TrainableScalarFunction;

public class ConvolutionalFilter implements TrainableScalarFunction<Float> {

    private final ConvolutionalShape shape;
    private final ConvolutionalKernel kernel;

    public static ConvolutionalFilter create(int size, int channels) {
        return new ConvolutionalFilter(new ConvolutionalShape(size, size, channels), new ConvolutionalKernel(Shape2D.of(size, size)));
    }

    public ConvolutionalFilter(ConvolutionalShape shape, ConvolutionalKernel kernel) {
        this.shape = shape;
        this.kernel = kernel;
    }

    public ConvolutionalShape getShape() {
        return shape;
    }
}
