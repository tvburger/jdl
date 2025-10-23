package net.tvburger.jdl.common.function;

import net.tvburger.jdl.common.numbers.Scalar;
import net.tvburger.jdl.common.patterns.Strategy;

@Strategy(Strategy.Role.INTERFACE)
public interface UnaryFunction<I, O> extends ScalarFunction<Scalar<I>, I, O> {

    default int arity() {
        return 1;
    }

}
