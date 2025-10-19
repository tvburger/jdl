package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Decorator;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.HyperparameterConfigurable;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * An abstract {@link Regime} implementation that delegates all training
 * to another {@link Regime}.
 *
 * <p>This class provides the common base for regime decorators such as
 * {@code BatchRegime}, {@code OnlineRegime}, or reporting regimes.
 * Subclasses can override {@link #train} to add cross-cutting behavior
 * (logging, early stopping, etc.) while still forwarding the actual
 * training call to the delegated regime.</p>
 *
 * <h3>Default behavior</h3>
 * <ul>
 *   <li>If a delegate regime is provided in the constructor, all calls are
 *   forwarded to it.</li>
 *   <li>If no delegate is provided ({@code null}), a default inline regime is
 *   created that simply invokes
 *   {@link Optimizer#optimize(net.tvburger.jdl.model.training.TrainableFunction, DataSet, ObjectiveFunction, int)}
 *   once on the entire dataset, i.e. a <em>single batch update</em>.</li>
 * </ul>
 */
@Decorator
public abstract class DelegatedRegime implements Regime, HyperparameterConfigurable {

    private final Map<String, Object> hyperparameters = new HashMap<>();

    /**
     * The regime that receives all delegated training calls.
     * Guaranteed to be non-null: if no delegate is provided,
     * a default batch regime is created internally.
     */
    protected final Regime regime;

    /**
     * Creates a new delegated regime.
     *
     * @param regime the regime to delegate to; if {@code null},
     *               a default regime is used that trains once on the
     *               full dataset via the given {@link Optimizer}.
     */
    protected DelegatedRegime(Regime regime) {
        this.regime = Objects.requireNonNull(regime);
    }

    /**
     * Returns the underlying delegated regime.
     *
     * @return the non-null delegate regime, either the one passed in the
     * constructor or the default batch regime
     */
    public Regime getDelegatedRegime() {
        return regime;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasHyperparameter(String name) {
        return hyperparameters.containsKey(name) || regime instanceof HyperparameterConfigurable configurable && configurable.hasHyperparameter(name);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getHyperparameters() {
        return hyperparameters;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setHyperparameter(String name, Object value) {
        if (!hyperparameters.containsKey(name) && regime instanceof HyperparameterConfigurable configurable && configurable.hasHyperparameter(name)) {
            configurable.setHyperparameter(name, value);
        } else {
            hyperparameters.put(name, value);
        }
    }
}
