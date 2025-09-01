package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Proxy;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * Composes and executes a chain of {@link Regime} decorators.
 *
 * <p>The chain stores regimes in order and, when trained, delegates to the
 * <em>top</em> (last) regime. If the chain is empty, this class will invoke
 * the provided {@link Optimizer} directly on the full {@link DataSet} as a
 * single batch (a safe no-op fallback if the caller omitted regimes).</p>
 *
 * <h3>Semantics</h3>
 * <ul>
 *   <li><strong>Top regime:</strong> the last element in {@code chainedRegimes} receives
 *       the {@link #train} call and typically wraps the rest.</li>
 *   <li><strong>Lookup:</strong> {@link #getRegime(Class)} and {@link #findRegime(java.util.function.Predicate)}
 *       return the first matching regime for introspection or reconfiguration.</li>
 *   <li><strong>Iteration:</strong> Implements {@link Iterable} to expose the chain order (top to bottom).</li>
 * </ul>
 */
@Strategy(Strategy.Role.CONCRETE)
public class ChainedRegime implements Regime, Iterable<Regime> {

    /**
     * The ordered chain of regimes; the last element is the top/outermost decorator.
     */
    protected final List<Regime> chainedRegimes;

    /**
     * Creates a new chain.
     *
     * @param chainedRegimes ordered regimes; the last is considered the "top".
     *                       May be empty; in that case {@link #train} falls back
     *                       to direct optimizer invocation if provided.
     */
    protected ChainedRegime(List<Regime> chainedRegimes) {
        this.chainedRegimes = chainedRegimes;
    }

    /**
     * Trains by delegating to the topmost regime in the chain. If the chain is empty,
     * invokes the {@link Optimizer} directly (if non-null) on the full dataset.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @param objective          the objective (loss) function
     * @param optimizer          the optimizer to apply updates (used directly if chain is empty)
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        if (chainedRegimes.isEmpty()) {
            if (optimizer != null) {
                optimizer.optimize(estimationFunction, trainingSet, objective);
            }
        } else {
            chainedRegimes.getLast().train(estimationFunction, trainingSet, objective, optimizer);
        }
    }

    /**
     * Returns an iterator over the regimes in from top to bottom.
     */
    @Override
    public Iterator<Regime> iterator() {
        return chainedRegimes.reversed().iterator();
    }

    /**
     * Returns the first regime assignable to {@code regimeTypeClass}, or throws if none.
     *
     * @param regimeTypeClass the desired regime type
     * @param <R>             type parameter of the regime
     * @return the first matching regime
     * @throws java.util.NoSuchElementException if none found
     */
    public <R extends Regime> R asRegime(Class<R> regimeTypeClass) {
        return getRegime(regimeTypeClass).orElseThrow();
    }

    /**
     * Returns the first regime assignable to the requested type, if any.
     *
     * @param regimeTypeClass the desired regime type
     * @param <R>             type parameter of the regime
     * @return an {@link Optional} containing the first match, or empty
     */
    public <R extends Regime> Optional<R> getRegime(Class<R> regimeTypeClass) {
        for (Regime regime : chainedRegimes) {
            if (regimeTypeClass.isInstance(regime)) {
                return Optional.of(regimeTypeClass.cast(regime));
            }
        }
        return Optional.empty();
    }

    /**
     * Returns the first regime matching the predicate, if any.
     *
     * @param predicate match condition
     * @return an {@link Optional} containing the first match, or empty
     */
    public Optional<Regime> findRegime(Predicate<? super Regime> predicate) {
        for (Regime regime : chainedRegimes) {
            if (predicate.test(regime)) return Optional.of(regime);
        }
        return Optional.empty();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "ChainedRegime" + chainedRegimes;
    }

    /**
     * A lightweight link that forwards {@link Regime#train} to another regime in the chain
     * by index (supports negative indexes counted from the end).
     *
     * <p>Used internally by builders to target the current top or relative positions
     * while the chain is being assembled.</p>
     *
     * @param chainedRegimes    Reference to the chained regimes
     * @param linkedRegimeIndex The target index; negative values are relative to the end.
     */
    @Proxy
    private record Link(List<Regime> chainedRegimes, int linkedRegimeIndex) implements Regime {

        /**
         * Resolves the linked regime by index (handling negative indexes).
         *
         * @return the target regime or {@code null} if out of bounds
         */
        private Regime getLinkedRegime() {
            int index = linkedRegimeIndex < 0 ? chainedRegimes.size() + linkedRegimeIndex : linkedRegimeIndex;
            return index < chainedRegimes.size() && index >= 0
                    ? chainedRegimes.get(index)
                    : null;
        }

        /**
         * Forwards training to the linked regime if available; otherwise falls back
         * to direct optimizer invocation (if non-null).
         */
        @Override
        public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
            Regime linkedRegime = getLinkedRegime();
            if (linkedRegime != null) {
                linkedRegime.train(estimationFunction, trainingSet, objective, optimizer);
            } else {
                if (optimizer != null) {
                    optimizer.optimize(estimationFunction, trainingSet, objective);
                }
            }
        }
    }

    @net.tvburger.jdl.common.patterns.Builder
    public static class Builder {

        private final List<Regime> chainedRegimes = new ArrayList<>();
        private Regime targetRegime = nextLink();

        private Link nextLink() {
            return new Link(chainedRegimes, -2 - chainedRegimes.size());
        }

        /**
         * Adds the given regime to the chain and updates {@link #targetRegime}.
         *
         * @param regime the regime to add
         * @return {@code this} builder
         */
        private Builder chainRegime(Regime regime) {
            chainedRegimes.addFirst(regime);
            targetRegime = nextLink();
            return this;
        }

        /**
         * Adds a {@link BatchRegime} decorator around the current target.
         *
         * @return this builder (for fluent chaining)
         * @see BatchRegime
         */
        public final ChainedRegime batch() {
            return chainRegime(new BatchRegime()).build();
        }

        /**
         * Adds an {@link OnlineRegime} decorator around the current target.
         *
         * @return this builder (for fluent chaining)
         * @see OnlineRegime
         */
        public final ChainedRegime online() {
            return chainRegime(new OnlineRegime()).build();
        }

        /**
         * Adds a {@link MiniBatchRegime} decorator with a default batch size of 32.
         *
         * @return this builder (for fluent chaining)
         * @see MiniBatchRegime
         */
        public final ChainedRegime miniBatch() {
            return miniBatch(32);
        }

        /**
         * Adds a {@link MiniBatchRegime} decorator with the given batch size.
         *
         * @param batchSize the number of samples per mini-batch (must be > 0)
         * @return this builder (for fluent chaining)
         * @see MiniBatchRegime
         */
        public final ChainedRegime miniBatch(int batchSize) {
            return chainRegime(new MiniBatchRegime(batchSize)).build();
        }

        /**
         * Adds an {@link EpochRegime} decorator with a default of 1,000 epochs.
         *
         * @return this builder (for fluent chaining)
         * @see EpochRegime
         */
        public final Builder epochs() {
            return epochs(1_000);
        }

        /**
         * Adds an {@link EpochRegime} decorator with the given number of epochs.
         *
         * @param epochs the number of epochs to run (must be >= 1)
         * @return this builder (for fluent chaining)
         * @see EpochRegime
         */
        public final Builder epochs(int epochs) {
            return chainRegime(new EpochRegime(targetRegime, epochs));
        }

        /**
         * Adds an {@link ObjectiveReportingRegime} decorator that reports
         * objective (loss) values with dumping enabled.
         *
         * @return this builder (for fluent chaining)
         * @see ObjectiveReportingRegime
         */
        public final Builder reportObjective() {
            return chainRegime(new ObjectiveReportingRegime(targetRegime, true));
        }

        /**
         * Adds a {@link DumpNodesRegime} decorator with default options (dump after training).
         *
         * @return this builder (for fluent chaining)
         * @see DumpNodesRegime
         */
        public final Builder dumpNodes() {
            return dumpNodes(false);
        }

        /**
         * Adds a {@link DumpNodesRegime} decorator with an initial dump toggle.
         *
         * @param firstTime whether to dump before the first training step
         * @return this builder (for fluent chaining)
         * @see DumpNodesRegime
         */
        public final Builder dumpNodes(boolean firstTime) {
            return dumpNodes(firstTime, false);
        }

        /**
         * Adds a {@link DumpNodesRegime} decorator with full configuration.
         *
         * @param firstTime     whether to dump before the first training step
         * @param includeInputs whether to include input nodes in the dump
         * @return this builder (for fluent chaining)
         * @see DumpNodesRegime
         */
        public final Builder dumpNodes(boolean firstTime, boolean includeInputs) {
            return chainRegime(new DumpNodesRegime(targetRegime, firstTime, includeInputs));
        }

        /**
         * Finalizes the builder and creates an immutable {@link ChainedRegime}.
         *
         * <p>The returned {@link ChainedRegime} contains the current list of
         * decorators in the order they were added. The underlying list is wrapped
         * with {@link List#copyOf(java.util.Collection)} to ensure immutability,
         * so further modifications to this builder will not affect the built chain.</p>
         *
         * @return an immutable {@link ChainedRegime} representing the current builder state
         * @see ChainedRegime
         */
        protected final ChainedRegime build() {
            List<Regime> copyChainedRegimes = new ArrayList<>(chainedRegimes);
            for (Regime regime : chainedRegimes) {
                if (regime instanceof Link link) {
                    copyChainedRegimes.add(new Link(copyChainedRegimes, link.linkedRegimeIndex));
                } else {
                    copyChainedRegimes.add(regime);
                }
            }
            return new ChainedRegime(copyChainedRegimes);
        }
    }

}
