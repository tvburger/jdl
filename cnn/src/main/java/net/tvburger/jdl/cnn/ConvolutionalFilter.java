package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.shapes.Shape2D;
import net.tvburger.jdl.model.scalars.TrainableScalarFunction;

public class ConvolutionalFilter implements TrainableScalarFunction<Float> {

    private final ConvolutionalKernel kernel;
    private Array<Float> parameters;

    public static ConvolutionalFilter create(int size, int channels) {
        Array<Float> parameters = JavaNumberTypeSupport.FLOAT.createArray(size * size * channels + 1);
        return new ConvolutionalFilter(new ConvolutionalKernel(Shape2D.of(size, size), channels, 1, parameters.slice(1)), parameters);
    }

    public ConvolutionalFilter(ConvolutionalKernel kernel, Array<Float> parameters) {
        this.kernel = kernel;
        this.parameters = parameters;
    }

    public ConvolutionalShape getShape() {
        return kernel;
    }

    public Float applyToWindow(ConvolutionalShape window) {
        return estimateScalar(window.getElements());
    }

    @Override
    public Array<Float> calculateParameterGradients(Array<Float> inputs) {
        return inputs;
    }

    @Override
    public Float estimateScalar(Array<Float> inputs) {
        Array<Float> weights = kernel.getWeights();
        float result = parameters.get(0);
        for (int i = 0; i < weights.length(); i++) {
            result += weights.get(i) * inputs.get(i);
        }
        return result;
    }

    @Override
    public int getParameterCount() {
        return parameters.length();
    }

    @Override
    public Array<Float> getParameters() {
        return parameters;
    }

    @Override
    public Float getParameter(int p) {
        return parameters.get(p);
    }

    @Override
    public void setParameters(Array<Float> values) {
        this.parameters = values;
    }

    @Override
    public void setParameter(int p, Float value) {
        parameters.set(p, value);
    }

    @Override
    public int arity() {
        return kernel.getElements().length();
    }

    @Override
    public JavaNumberTypeSupport<Float> getNumberTypeSupport() {
        return JavaNumberTypeSupport.FLOAT;
    }
}
