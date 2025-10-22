package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.shapes.Shape2D;
import net.tvburger.jdl.model.Kernel;

public class ConvolutionalKernel extends ConvolutionalShape implements Kernel {

    private final Shape2D kernelSize;
    private final Array<Float> weights;

    public ConvolutionalKernel(Shape2D kernelSize, int channels, int count, Array<Float> weights) {
        super(kernelSize.getWidth(), kernelSize.getHeight(), channels, count);
        this.kernelSize = kernelSize;
        this.weights = weights;
    }

    public Shape2D getKernelSize() {
        return kernelSize;
    }

    public Array<Float> getWeights() {
        return weights;
    }
}
