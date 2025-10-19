package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.StaticUtility;
import net.tvburger.jdl.model.training.Regime;

/**
 * Static factory methods for creating and configuring common
 * {@link Regime} strategies.
 * <p>
 * This utility class provides convenient shortcuts to construct
 * standard training regimes such as full-batch, mini-batch, or
 * online training, as well as builders for chained regimes with
 * additional behaviors (e.g., multiple epochs, reporting, or
 * dumping neuron states).
 */
@StaticUtility
public final class Regimes {

    private Regimes() {
    }

    public static OneShotRegime oneShot() {
        return new OneShotRegime();
    }

    /**
     * Creates a full-batch training regime that processes the entire dataset
     * in a single update.
     *
     * @return a new {@link BatchRegime}
     */
    public static BatchRegime batch() {
        return new BatchRegime();
    }

    /**
     * Creates a mini-batch training regime with the given batch size.
     *
     * @param batchSize the number of samples per mini-batch (must be &gt; 0)
     * @return a new {@link MiniBatchRegime}
     */
    public static MiniBatchRegime miniBatch(int batchSize) {
        return new MiniBatchRegime(batchSize);
    }

    /**
     * Creates an online training regime that processes one sample at a time.
     *
     * @return a new {@link StochasticRegime}
     */
    public static StochasticRegime online() {
        return new StochasticRegime();
    }

    /**
     * Creates an online training regime that processes one sample at a time.
     *
     * @return a new {@link StochasticRegime}
     */
    public static StochasticRegime stochastic() {
        return new StochasticRegime();
    }

    /**
     * Starts building a chained regime that repeats training for a given
     * number of epochs.
     *
     * @param epochs the number of epochs (must be &gt; 0)
     * @return a {@link ChainedRegime.Builder} preconfigured with epoch count
     */
    public static ChainedRegime.Builder epochs(int epochs) {
        return new ChainedRegime.Builder().epochs(epochs);
    }

    public static ChainedRegime.Builder epochs(int epochs, EpochRegime.EpochCompletionListener... listener) {
        return new ChainedRegime.Builder().epochs(epochs, listener);
    }

    /**
     * Starts building a chained regime that reports the objective function
     * after training.
     *
     * @return a {@link ChainedRegime.Builder} preconfigured to report the objective
     */
    public static ChainedRegime.Builder reportObjective() {
        return new ChainedRegime.Builder().reportObjective();
    }

    /**
     * Starts building a chained regime that dumps neuron states after training.
     *
     * @return a {@link ChainedRegime.Builder} preconfigured to dump nodes
     */
    public static ChainedRegime.Builder dumpNodes() {
        return new ChainedRegime.Builder().dumpNodes();
    }

    /**
     * Starts building a chained regime that dumps neuron states, optionally
     * only during the first pass.
     *
     * @param firstTime if {@code true}, dump nodes only once
     * @return a {@link ChainedRegime.Builder} preconfigured to dump nodes
     */
    public static ChainedRegime.Builder dumpNodes(boolean firstTime) {
        return new ChainedRegime.Builder().dumpNodes(firstTime);
    }

    /**
     * Starts building a chained regime that dumps neuron states, optionally
     * only during the first pass and optionally including input neurons.
     *
     * @param firstTime     if {@code true}, dump nodes only once
     * @param includeInputs if {@code true}, include input nodes in the dump
     * @return a {@link ChainedRegime.Builder} preconfigured to dump nodes
     */
    public static ChainedRegime.Builder dumpNodes(boolean firstTime, boolean includeInputs) {
        return new ChainedRegime.Builder().dumpNodes(firstTime, includeInputs);
    }

}