package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.function.ScalarFunction;
import net.tvburger.jdl.common.numbers.Array;

public interface PoolingFunction<N> extends ScalarFunction<N, N> {

    default N mapToScalar(Array<N> input) {
        return pool(input);
    }

    N pool(Array<N> elements);

}
