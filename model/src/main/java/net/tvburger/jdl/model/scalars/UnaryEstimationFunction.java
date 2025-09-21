package net.tvburger.jdl.model.scalars;

public interface UnaryEstimationFunction extends ScalarEstimationFunction {

    @Override
    default float estimateScalar(float[] inputs) {
        if (inputs.length != 1) {
            throw new IllegalArgumentException("Invalid arity!");
        }
        return estimateUnary(inputs[0]);
    }

    float estimateUnary(float input);

    @Override
    default int arity() {
        return 1;
    }

}
