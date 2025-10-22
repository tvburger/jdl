package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.numbers.Array;
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
    public N calculateLossWithoutRegularizationPenalty(List<Pair<Array<N>, Array<N>>> batch) {
        List<N> sampleLosses = new ArrayList<>(batch.size());
        List<N> dimensionLosses = new ArrayList<>();
        for (Pair<Array<N>, Array<N>> sample : batch) {
            Array<N> estimated = sample.left();
            Array<N> target = sample.right();
            int dimensions = estimated.length();
            for (int d = 0; d < dimensions; d++) {
                dimensionLosses.add(getDimensionLossFunction(d).calculateDimensionLoss(estimated.get(d), target.get(d)));
            }
            sampleLosses.add(sampleLossFunction.calculateSampleLoss(dimensionLosses));
            dimensionLosses.clear();
        }
        return batchLossFunction.calculateBatchLoss(sampleLosses);
    }

    @Override
    public N calculateRegularizationPenalty(Array<N> parameters) {
        N totalPenalty = getNumberTypeSupport().zero();
        for (ExplicitRegularization<N> regularization : regularizations) {
            totalPenalty = getNumberTypeSupport().add(totalPenalty, regularization.lossPenalty(parameters));
        }
        return totalPenalty;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array<N> calculateGradient_dJ_da(int batchSize, Array<N> estimated, Array<N> target) {
        int dimensions = target.length();
        JavaNumberTypeSupport<N> typeSupport = getNumberTypeSupport();
        Array<N> gradients = typeSupport.createArray(dimensions);
        for (int d = 0; d < dimensions; d++) {
            gradients.set(d, typeSupport.multiply(typeSupport.multiply(
                            batchLossFunction.calculateGradient_dJ_dL(batchSize),
                            sampleLossFunction.calculateGradient_dL_dl(dimensions)),
                    getDimensionLossFunction(d).calculateGradient_dl_da(estimated.get(d), target.get(d))));
        }
        return gradients;
    }

    @Override
    public N regularizedGradient(N parameter) {
        N totalAdjustment = getNumberTypeSupport().zero();
        for (ExplicitRegularization<N> regularization : regularizations) {
            totalAdjustment = getNumberTypeSupport().add(totalAdjustment, regularization.gradientAdjustment(parameter));
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
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return batchLossFunction.getNumberTypeSupport();
    }
}
