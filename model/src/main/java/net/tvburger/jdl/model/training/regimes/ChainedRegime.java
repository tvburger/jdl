package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Builder;
import net.tvburger.jdl.common.patterns.Proxy;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.patterns.TemplateMethod;
import net.tvburger.jdl.common.utils.Floats;
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
@Strategy(role = Strategy.Role.CONCRETE)
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
     */
    @Proxy
    private class Link implements Regime {

        /**
         * The target index; negative values are relative to the end.
         */
        private final int linkedRegimeIndex;

        private Link(int linkedRegimeIndex) {
            this.linkedRegimeIndex = linkedRegimeIndex;
        }

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

    /**
     * Abstract base for creating fluent builders of {@link ChainedRegime}.
     *
     * <p>This class is not used directly; instead, concrete subclasses such as
     * {@link TopDownChainer} and {@link BottomUpChainer} extend it to provide
     * a fluent DSL for assembling training regimes. It maintains a list of
     * chained regimes and a {@code targetRegime} pointer so that each newly
     * added decorator can wrap the current top of the chain.</p>
     *
     * <p>Fluent methods such as {@link #epochs(int)}, {@link #miniBatch(int)},
     * and {@link #reportObjective()} add specific {@link Regime} decorators
     * around the current target, enabling expressive construction of complex
     * training pipelines without manual wiring.</p>
     *
     * <p>Once the chain is complete, {@link #build()} produces an immutable
     * {@link ChainedRegime} that can be passed to a {@code Trainer}.</p>
     *
     * @param <M> the concrete builder subtype (for fluent chaining)
     * @see TopDownChainer
     * @see BottomUpChainer
     */
    @TemplateMethod
    protected static abstract class Modifiable<M extends Modifiable<M>> extends ChainedRegime {

        /**
         * The regime that the next decorator should wrap.
         */
        protected Regime targetRegime;

        /**
         * Creates a new, initially empty chain builder.
         */
        protected Modifiable() {
            super(new ArrayList<>());
        }

        /**
         * Adds the given regime to the chain and updates {@link #targetRegime}.
         *
         * @param regime the regime to add
         * @return {@code this} builder
         */
        protected abstract M chainRegime(Regime regime);

        /**
         * Adds a {@link BatchRegime} decorator around the current target.
         *
         * @return this builder (for fluent chaining)
         * @see BatchRegime
         */
        public final M batch() {
            return chainRegime(new BatchRegime(targetRegime));
        }

        /**
         * Adds an {@link OnlineRegime} decorator around the current target.
         *
         * @return this builder (for fluent chaining)
         * @see OnlineRegime
         */
        public final M online() {
            return chainRegime(new OnlineRegime(targetRegime));
        }

        /**
         * Adds a {@link MiniBatchRegime} decorator with a default batch size of 32.
         *
         * @return this builder (for fluent chaining)
         * @see MiniBatchRegime
         */
        public final M miniBatch() {
            return miniBatch(32);
        }

        /**
         * Adds a {@link MiniBatchRegime} decorator with the given batch size.
         *
         * @param samplesPerLearning the number of samples per mini-batch (must be > 0)
         * @return this builder (for fluent chaining)
         * @see MiniBatchRegime
         */
        public final M miniBatch(int samplesPerLearning) {
            return chainRegime(new MiniBatchRegime(targetRegime, samplesPerLearning));
        }

        /**
         * Adds an {@link EpochRegime} decorator with a default of 1,000 epochs.
         *
         * @return this builder (for fluent chaining)
         * @see EpochRegime
         */
        public final M epochs() {
            return epochs(1_000);
        }

        /**
         * Adds an {@link EpochRegime} decorator with the given number of epochs.
         *
         * @param epochs the number of epochs to run (must be >= 1)
         * @return this builder (for fluent chaining)
         * @see EpochRegime
         */
        public final M epochs(int epochs) {
            return chainRegime(new EpochRegime(targetRegime, epochs));
        }


        /**
         * Adds an {@link ObjectiveReportingRegime} decorator that reports
         * objective (loss) values with dumping enabled.
         *
         * @return this builder (for fluent chaining)
         * @see ObjectiveReportingRegime
         */
        public final M reportObjective() {
            return chainRegime(new ObjectiveReportingRegime(targetRegime, true));
        }

        /**
         * Adds a {@link StopIfNoImprovementRegime} decorator with default
         * parameters (stall limit = 2, min improvement = {@link Floats#EPSILON}).
         *
         * @return this builder (for fluent chaining)
         * @see StopIfNoImprovementRegime
         */
        public final M stopIfNoImprovements() {
            return stopIfNoImprovements(2);
        }

        /**
         * Adds a {@link StopIfNoImprovementRegime} decorator with the given stall limit.
         *
         * @param maxStalledEpochs maximum consecutive stalled epochs allowed
         * @return this builder (for fluent chaining)
         * @see StopIfNoImprovementRegime
         */
        public final M stopIfNoImprovements(int maxStalledEpochs) {
            return stopIfNoImprovements(maxStalledEpochs, Floats.EPSILON);
        }

        /**
         * Adds a {@link StopIfNoImprovementRegime} decorator with default stall limit (2)
         * and the specified dumping flag.
         *
         * @param dumpLosses whether to print loss values during training
         * @return this builder (for fluent chaining)
         * @see StopIfNoImprovementRegime
         */
        public final M stopIfNoImprovements(boolean dumpLosses) {
            return stopIfNoImprovements(2, dumpLosses);
        }

        /**
         * Adds a {@link StopIfNoImprovementRegime} decorator with a stall limit
         * and dumping flag.
         *
         * @param maxStalledEpochs maximum consecutive stalled epochs allowed
         * @param dumpLosses       whether to print loss values during training
         * @return this builder (for fluent chaining)
         * @see StopIfNoImprovementRegime
         */
        public final M stopIfNoImprovements(int maxStalledEpochs, boolean dumpLosses) {
            return stopIfNoImprovements(2, Floats.EPSILON, dumpLosses);
        }


        /**
         * Adds a {@link StopIfNoImprovementRegime} decorator with the given
         * stall limit and minimum improvement threshold.
         *
         * @param maxStalledEpochs maximum consecutive stalled epochs allowed
         * @param minImprovement   minimum relative improvement (percentage) required
         * @return this builder (for fluent chaining)
         * @see StopIfNoImprovementRegime
         */
        public final M stopIfNoImprovements(int maxStalledEpochs, float minImprovement) {
            return stopIfNoImprovements(maxStalledEpochs, minImprovement, true);
        }

        /**
         * Adds a {@link StopIfNoImprovementRegime} decorator with full configuration.
         *
         * @param maxStalledEpochs maximum consecutive stalled epochs allowed
         * @param minImprovement   minimum relative improvement (percentage) required
         * @param dumpLosses       whether to print loss values during training
         * @return this builder (for fluent chaining)
         * @see StopIfNoImprovementRegime
         */
        public final M stopIfNoImprovements(int maxStalledEpochs, float minImprovement, boolean dumpLosses) {
            return chainRegime(StopIfNoImprovementRegime.create(targetRegime, maxStalledEpochs, minImprovement, dumpLosses));
        }

        /**
         * Adds a {@link DumpNodesRegime} decorator with default options (dump after training).
         *
         * @return this builder (for fluent chaining)
         * @see DumpNodesRegime
         */
        public final M dumpNodes() {
            return dumpNodes(false);
        }

        /**
         * Adds a {@link DumpNodesRegime} decorator with an initial dump toggle.
         *
         * @param firstTime whether to dump before the first training step
         * @return this builder (for fluent chaining)
         * @see DumpNodesRegime
         */
        public final M dumpNodes(boolean firstTime) {
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
        public final M dumpNodes(boolean firstTime, boolean includeInputs) {
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
            return new ChainedRegime(List.copyOf(chainedRegimes));
        }

    }

    /**
     * Fluent builder that composes the chain from <em>top down</em>.
     * <p>Each added decorator becomes the new head, wrapping the previous head.
     * Internally uses a {@link Link} to refer to the evolving predecessor.</p>
     */
    @Builder
    public static final class TopDownChainer extends ChainedRegime.Modifiable<TopDownChainer> {

        /**
         * Initializes the builder with a link pointing one-before-last, then moves as items are added.
         */
        public TopDownChainer() {
            targetRegime = nextLink();
        }

        private Link nextLink() {
            return new Link(-2 - chainedRegimes.size());
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected TopDownChainer chainRegime(Regime regime) {
            chainedRegimes.addFirst(regime);
            targetRegime = nextLink();
            return this;
        }

        /**
         * Finalizes the composition and returns unmodifiable chain.
         */
        public ChainedRegime bottomChain() {
            return build();
        }
    }

    /**
     * Fluent builder that composes the chain from <em>bottom up</em>.
     * <p>Each added decorator wraps the previous target; the most recently added
     * decorator becomes the top.</p>
     */
    @Builder
    public static final class BottomUpChainer extends ChainedRegime.Modifiable<BottomUpChainer> {

        /**
         * {@inheritDoc}
         */
        @Override
        protected BottomUpChainer chainRegime(Regime regime) {
            targetRegime = regime;
            chainedRegimes.add(targetRegime);
            return this;
        }

        /**
         * Finalizes the composition and returns unmodifiable chain.
         */
        public ChainedRegime topChain() {
            return build();
        }

    }

}
