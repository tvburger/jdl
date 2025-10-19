package net.tvburger.jdl.model.training.loss;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.StaticUtility;
import net.tvburger.jdl.model.training.ObjectiveFunction;

/**
 * Provides commonly used {@link ObjectiveFunction} implementations as
 * reusable, static singletons.
 *
 * <p>
 * The {@code Losses} class serves as a convenience utility for retrieving
 * standard objective functions such as Mean Squared Error (MSE) and Binary
 * Cross-Entropy (BCE). These functions are frequently used in optimization
 * and machine learning tasks, making them natural defaults for training
 * pipelines.
 */
@StaticUtility
public final class Objectives {

    private Objectives() {
    }

    /**
     * Returns the shared Mean Squared Error (MSE) objective function.
     *
     * @return the predefined MSE {@link ObjectiveFunction}
     */
    public static <N extends Number> ObjectiveFunction<N> mSE(JavaNumberTypeSupport<N> typeSupport) {
        N scale = typeSupport.divide(typeSupport.one(), typeSupport.add(typeSupport.one(), typeSupport.one()));
        return ObjectiveFunction.minimize(new MeanError<>(typeSupport), new MeanError<>(typeSupport), new ScaledError<>(typeSupport, scale, new SquaredError<>(typeSupport)));
    }

    /**
     * Returns the shared Summed Squared Error (SSE) objective function.
     *
     * @return the predefined SSE {@link ObjectiveFunction}
     */
    public static <N extends Number> ObjectiveFunction<N> sSE(JavaNumberTypeSupport<N> typeSupport) {
        N scale = typeSupport.divide(typeSupport.one(), typeSupport.add(typeSupport.one(), typeSupport.one()));
        return ObjectiveFunction.minimize(new SummedError<>(typeSupport), new SummedError<>(typeSupport), new ScaledError<>(typeSupport, scale, new SquaredError<>(typeSupport)));
    }

    /**
     * Returns the shared Binary Cross-Entropy (BCE) objective function.
     *
     * @return the predefined BCE {@link ObjectiveFunction}
     */
    public static <N extends Number> ObjectiveFunction<N> bCE(JavaNumberTypeSupport<N> typeSupport) {
        return ObjectiveFunction.minimize(new MeanError<>(typeSupport), new MeanError<>(typeSupport), new BinaryCrossEntropy<>(typeSupport));
    }
}
