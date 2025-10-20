package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
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

    @Override
    public Float[] calculateParameterGradients(Float[] inputs) {
        return new Float[0];
    }

    @Override
    public Float estimateScalar(Float[] inputs) {
        return 0f;
    }

    @Override
    public int getParameterCount() {
        return 0;
    }

    @Override
    public Float[] getParameters() {
        return new Float[0];
    }

    @Override
    public Float getParameter(int p) {
        return 0f;
    }

    @Override
    public void setParameters(Float[] values) {

    }

    @Override
    public void setParameter(int p, Float value) {

    }

    @Override
    public int arity() {
        return 0;
    }

    @Override
    public JavaNumberTypeSupport<Float> getCurrentNumberType() {
        return null;
    }
}
