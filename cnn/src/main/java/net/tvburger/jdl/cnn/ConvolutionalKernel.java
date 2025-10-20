package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.shapes.Shape2D;
import net.tvburger.jdl.model.Kernel;
import net.tvburger.jdl.model.scalars.TrainableScalarFunction;

public class ConvolutionalKernel implements Kernel, TrainableScalarFunction<Float> {

    private final Shape2D kernelSize;

    public ConvolutionalKernel(Shape2D kernelSize) {
        this.kernelSize = kernelSize;
    }

    public Shape2D getKernelSize() {
        return kernelSize;
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
