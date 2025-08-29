package net.tvburger.jdl.model.training.loss;

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
public final class Losses {

    /**
     * Predefined Mean Squared Error (MSE) objective function.
     *
     * <p>
     * Internally composed of:
     * <ul>
     *   <li>{@link MeanError} for batch-level aggregation</li>
     *   <li>{@link MeanError} for aggregating sample errors</li>
     *   <li>{@link ScaledError} with factor {@code 0.5} wrapping a
     *       {@link SquaredError} for dimension-level error</li>
     * </ul>
     * </p>
     */
    private static final ObjectiveFunction mse = ObjectiveFunction.minimize(new MeanError(), new MeanError(), new ScaledError(0.5f, new SquaredError()));

    /**
     * Predefined Summed Squared Error (SSE) objective function.
     *
     * <p>
     * Internally composed of:
     * <ul>
     *   <li>{@link SummedError} for batch-level aggregation</li>
     *   <li>{@link SummedError} for aggregating sample errors</li>
     *   <li>{@link ScaledError} with factor {@code 0.5} wrapping a
     *       {@link SquaredError} for dimension-level error</li>
     * </ul>
     * </p>
     */
    private static final ObjectiveFunction sse = ObjectiveFunction.minimize(new SummedError(), new SummedError(), new ScaledError(0.5f, new SquaredError()));

    /**
     * Predefined Binary Cross-Entropy (BCE) objective function.
     *
     * <p>
     * Internally composed of:
     * <ul>
     *   <li>{@link MeanError} for batch-level aggregation</li>
     *   <li>{@link MeanError} for aggregating sample errors</li>
     *   <li>{@link BinaryCrossEntropy} for dimension-level error</li>
     * </ul>
     * </p>
     */
    private static final ObjectiveFunction bce = ObjectiveFunction.minimize(new MeanError(), new MeanError(), new BinaryCrossEntropy());

    private Losses() {
    }

    /**
     * Returns the shared Mean Squared Error (MSE) objective function.
     *
     * @return the predefined MSE {@link ObjectiveFunction}
     */
    public static ObjectiveFunction mSE() {
        return mse;
    }

    /**
     * Returns the shared Summed Squared Error (SSE) objective function.
     *
     * @return the predefined SSE {@link ObjectiveFunction}
     */
    public static ObjectiveFunction sSE() {
        return mse;
    }

    /**
     * Returns the shared Binary Cross-Entropy (BCE) objective function.
     *
     * @return the predefined BCE {@link ObjectiveFunction}
     */
    public static ObjectiveFunction bCE() {
        return bce;
    }
}
