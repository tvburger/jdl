package net.tvburger.jdl.model.scalars;

import net.tvburger.jdl.common.numbers.Array;

public interface UnaryEstimationFunction<N extends Number> extends ScalarEstimationFunction<N> {

    @Override
    default N estimateScalar(Array<N> inputs) {
        if (inputs.length() != 1) {
            throw new IllegalArgumentException("Invalid arity!");
        }
        return estimateUnary(inputs.get(0));
    }

    N estimateUnary(N input);

    @Override
    default int arity() {
        return 1;
    }

}
