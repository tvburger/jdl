package net.tvburger.jdl.model.scalars;

public interface UnaryEstimationFunction<N extends Number> extends ScalarEstimationFunction<N> {

    @Override
    default N estimateScalar(N[] inputs) {
        if (inputs.length != 1) {
            throw new IllegalArgumentException("Invalid arity!");
        }
        return estimateUnary(inputs[0]);
    }

    N estimateUnary(N input);

    @Override
    default int arity() {
        return 1;
    }

}
