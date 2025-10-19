package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Mediator;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.model.training.loss.BatchLossFunction;
import net.tvburger.jdl.model.training.loss.DimensionLossFunction;
import net.tvburger.jdl.model.training.loss.SampleLossFunction;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;

import java.util.*;

/**
 * The implementation of the {@link ObjectiveFunction}.
 *
 * <p>
 * {@code ObjectiveFunctionImpl} composes together different levels of
 * loss functions — dimension, sample, and batch — to provide a complete
 * objective function suitable for training and optimization.
 * </p>
 *
 * <p>
 * The implementation delegates responsibility as follows:
 * <ul>
 *   <li>Uses a {@link DimensionLossFunction} to compute per-dimension error
 *       and parameterGradients.</li>
 *   <li>Uses a {@link SampleLossFunction} to aggregate dimension-level losses
 *       into a single sample loss.</li>
 *   <li>Uses a {@link BatchLossFunction} to aggregate sample losses into a
 *       batch loss.</li>
 * </ul>
 * </p>
 */
@Mediator
@Strategy(Strategy.Role.CONCRETE)
public class ObjectiveFunctionImpl<N extends Number> implements ObjectiveFunction<N> {

    private final Set<ExplicitRegularization<N>> regularizations = new LinkedHashSet<>();

    private final SampleLossFunction<N> sampleLossFunction;
    private final List<DimensionLossFunction<N>> dimensionLossFunctions;
    private final BatchLossFunction<N> batchLossFunction;
    private final boolean optimize;

    /**
     * Creates an {@code ObjectiveFunctionImpl} that computes losses at all levels:
     * dimension, sample, and batch.
     *
     * <p>This constructor links together the components needed for hierarchical
     * loss computation:</p>
     * <ul>
     *     <li>{@link BatchLossFunction}: computes the total loss over a batch of samples.</li>
     *     <li>{@link SampleLossFunction}: computes the loss for a single sample by
     *         aggregating per-dimension losses.</li>
     *     <li>{@link DimensionLossFunction}: computes the loss for each individual
     *         output dimension.</li>
     * </ul>
     *
     * @param batchLossFunction      the loss function for the batch level
     * @param sampleLossFunction     the loss function for the sample level
     * @param dimensionLossFunctions the list of loss functions for each output dimension
     */
    public ObjectiveFunctionImpl(BatchLossFunction<N> batchLossFunction, SampleLossFunction<N> sampleLossFunction, List<DimensionLossFunction<N>> dimensionLossFunctions, boolean optimize) {
        this.batchLossFunction = batchLossFunction;
        this.sampleLossFunction = sampleLossFunction;
        this.dimensionLossFunctions = dimensionLossFunctions;
        this.optimize = optimize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isOptimization() {
        return optimize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void addRegularization(ExplicitRegularization<N> regularization) {
        regularizations.add(regularization);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void removeRegularization(ExplicitRegularization<N> regularization) {
        regularizations.remove(regularization);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Set<ExplicitRegularization<N>> getRegularizations() {
        return Collections.unmodifiableSet(regularizations);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N calculateLossWithoutRegularizationPenalty(List<Pair<N[], N[]>> batch) {
        List<N> sampleLosses = new ArrayList<>(batch.size());
        List<N> dimensionLosses = new ArrayList<>();
        for (Pair<N[], N[]> sample : batch) {
            N[] estimated = sample.left();
            N[] target = sample.right();
            int dimensions = estimated.length;
            for (int d = 0; d < dimensions; d++) {
                dimensionLosses.add(getDimensionLossFunction(d).calculateDimensionLoss(estimated[d], target[d]));
            }
            sampleLosses.add(sampleLossFunction.calculateSampleLoss(dimensionLosses));
            dimensionLosses.clear();
        }
        return batchLossFunction.calculateBatchLoss(sampleLosses);
    }

    @Override
    public N calculateRegularizationPenalty(N[] parameters) {
        N totalPenalty = getCurrentNumberType().zero();
        for (ExplicitRegularization<N> regularization : regularizations) {
            totalPenalty = getCurrentNumberType().add(totalPenalty, regularization.lossPenalty(parameters));
        }
        return totalPenalty;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public N[] calculateGradient_dJ_da(int batchSize, N[] estimated, N[] target) {
        int dimensions = target.length;
        JavaNumberTypeSupport<N> typeSupport = getCurrentNumberType();
        N[] gradients = typeSupport.createArray(dimensions);
        for (int d = 0; d < dimensions; d++) {
            gradients[d] = typeSupport.multiply(typeSupport.multiply(
                            batchLossFunction.calculateGradient_dJ_dL(batchSize),
                            sampleLossFunction.calculateGradient_dL_dl(dimensions)),
                    getDimensionLossFunction(d).calculateGradient_dl_da(estimated[d], target[d]));
        }
        return gradients;
    }

    @Override
    public N regularizedGradient(N parameter) {
        N totalAdjustment = getCurrentNumberType().zero();
        for (ExplicitRegularization<N> regularization : regularizations) {
            totalAdjustment = getCurrentNumberType().add(totalAdjustment, regularization.gradientAdjustment(parameter));
        }
        return totalAdjustment;
    }

    private DimensionLossFunction<N> getDimensionLossFunction(int d) {
        return dimensionLossFunctions.get(d % dimensionLossFunctions.size());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public JavaNumberTypeSupport<N> getCurrentNumberType() {
        return batchLossFunction.getCurrentNumberType();
    }
}
