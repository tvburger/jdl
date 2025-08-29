package net.tvburger.jdl.model;

import net.tvburger.jdl.common.patterns.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Represents a data set. A data set consist of samples. A sample has inputs (e.g. features or signals) and
 * corresponding expected outputs.
 *
 * @param samples the samples to hold in this data set
 */
@DomainObject
@Composition
public record DataSet(List<Sample> samples) implements Iterable<DataSet.Sample> {

    /**
     * Interface to load a data set.
     */
    @FactoryMethod
    public interface Loader {

        /**
         * The loaded data set.
         *
         * @return the loaded data set
         */
        DataSet load();

    }

    /**
     * The sample represents a single input for an estimation function and the corresponding expected outputs.
     *
     * @param features      the inputs for this sample
     * @param targetOutputs the expected outputs for the given inputs
     * @see EstimationFunction
     */
    @DomainObject
    @ValueObject
    public record Sample(float[] features, float[] targetOutputs) {

        /**
         * Utility method to create a sample
         *
         * @param feature      the inputs for the sample
         * @param targetOutput the expected outputs for the sample
         * @return the sample
         */
        @StaticFactory
        public static DataSet.Sample of(float[] feature, float[] targetOutput) {
            return new DataSet.Sample(feature, targetOutput);
        }

        /**
         * Returns the number of features in this sample.
         * <p>
         * The feature count corresponds to the dimensionality of the
         * input vector (the <b>arity</b> of the function being trained or
         * estimated). This is typically the number of independent variables
         * or input attributes provided by the dataset.
         * </p>
         *
         * @return the number of input features in this sample
         */
        public int featureCount() {
            return features.length;
        }

        /**
         * Returns the number of target outputs in this sample.
         * <p>
         * The target count corresponds to the dimensionality of the
         * expected output vector (the <b>co-arity</b> of the function being
         * trained or estimated). This is typically the number of dependent
         * variables, labels, or regression targets provided by the dataset.
         * </p>
         *
         * @return the number of target outputs in this sample
         */
        public int targetCount() {
            return targetOutputs.length;
        }

        /**
         * Checks whether this sample is compatible with the given estimation function.
         * <p>
         * A dataset is considered compatible if:
         * <ul>
         *   <li>its feature count equals the function's {@code arity()}, and</li>
         *   <li>its target count equals the function's {@code coArity()}.</li>
         * </ul>
         *
         * @param estimationFunction the estimation function to compare against
         * @return {@code true} if the dataset matches the function's input/output schema;
         * {@code false} otherwise
         */
        public boolean isCompatibleWith(EstimationFunction estimationFunction) {
            return estimationFunction.arity() == featureCount() && estimationFunction.coArity() == targetCount();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Sample sample = (Sample) o;
            return Arrays.equals(features, sample.features) && Arrays.equals(targetOutputs, sample.targetOutputs);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public int hashCode() {
            int result = Arrays.hashCode(features);
            result = 31 * result + Arrays.hashCode(targetOutputs);
            return result;
        }
    }

    /**
     * Create a new data set for the given samples, where samples can't added or removed.
     *
     * @param sample the samples to put in the data set
     * @return the data set
     */
    @StaticFactory
    public static DataSet of(Sample... sample) {
        if (sample.length > 0) {
            Sample firstSample = sample[0];
            for (int i = 1; i < sample.length; i++) {
                if (firstSample.featureCount() != sample[i].featureCount()
                        || firstSample.targetCount() != sample[i].targetCount()) {
                    throw new IllegalArgumentException("Sample " + i + " has wrong size!");
                }
            }
        }
        return new DataSet(List.of(sample));
    }

    /**
     * Creates a new empty data set
     */
    @StaticFactory
    public DataSet() {
        this(new ArrayList<>());
    }

    /**
     * Creates a new data set that contains only a subset of samples. Useful for cutting a single data set into
     * a training, validation, and test set.
     *
     * @param fromIndex the fromIndex inclusive
     * @param toIndex   the toIndex exclusive
     * @return the new data set containing only a subset of the samples
     * @throws IndexOutOfBoundsException if the indexes are not properly given
     */
    public DataSet subset(int fromIndex, int toIndex) {
        return new DataSet(samples.subList(fromIndex, toIndex));
    }

    /**
     * Adds the given sample to the data set
     *
     * @param sample the sample to add
     */
    public void addSample(Sample sample) {
        if (!samples.isEmpty()) {
            if (sample.featureCount() != samples.getFirst().featureCount()
                    || sample.targetCount() != samples.getFirst().targetCount()) {
                throw new IllegalArgumentException("Sample has wrong size!");
            }
        }
        samples.add(sample);
    }

    /**
     * Adds the given sample to the data set
     *
     * @param features      the inputs for the sample
     * @param targetOutputs the expected outputs for the sample
     */
    public void addSample(float[] features, float[] targetOutputs) {
        addSample(DataSet.Sample.of(features, targetOutputs));
    }

    /**
     * Removes the first occurrence of the sample of the data set. No-op if not present.
     *
     * @param sample the sample to remove
     */
    public void removeSample(Sample sample) {
        samples.remove(sample);
    }

    /**
     * Returns the number of features in this data set.
     * <p>
     * The feature count corresponds to the dimensionality of the
     * input vector (the <b>arity</b> of the function being trained or
     * estimated). This is typically the number of independent variables
     * or input attributes provided by the dataset.
     * </p>
     *
     * @return the number of input features in this sample
     */
    public int getFeatureCount() {
        if (samples.isEmpty()) {
            throw new IllegalStateException("No samples!");
        }
        return samples.getFirst().featureCount();
    }

    /**
     * Returns the number of target outputs in this data set.
     * <p>
     * The target count corresponds to the dimensionality of the
     * expected output vector (the <b>co-arity</b> of the function being
     * trained or estimated). This is typically the number of dependent
     * variables, labels, or regression targets provided by the dataset.
     * </p>
     *
     * @return the number of target outputs in this sample
     */
    public int getTargetCount() {
        if (samples.isEmpty()) {
            throw new IllegalStateException("No samples!");
        }
        return samples.getFirst().targetCount();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Iterator<Sample> iterator() {
        return samples().listIterator();
    }

    /**
     * Checks whether this dataset is compatible with the given estimation function.
     * <p>
     * A dataset is considered compatible if:
     * <ul>
     *   <li>its feature count per sample equals the function's {@code arity()}, and</li>
     *   <li>its target count per sample equals the function's {@code coArity()}.</li>
     * </ul>
     *
     * @param estimationFunction the estimation function to compare against
     * @return {@code true} if the dataset matches the function's input/output schema;
     * {@code false} otherwise
     */
    public boolean isCompatibleWith(EstimationFunction estimationFunction) {
        return samples.isEmpty() || samples.getFirst().isCompatibleWith(estimationFunction);
    }

    /**
     * Ensures that this {@link DataSet} is compatible with the given
     * {@link EstimationFunction}.
     *
     * <p>
     * Compatibility typically means that the structure of the dataset
     * (e.g., feature dimensionality, output dimensionality) matches
     * the requirements of the estimation function. This method is a
     * guard to prevent training or evaluation with mismatched models
     * and datasets.
     * </p>
     *
     * <p>
     * If the dataset is compatible, this method returns the current
     * instance for fluent chaining. If it is not compatible, an
     * {@link IllegalArgumentException} is thrown.
     * </p>
     *
     * @param estimationFunction the estimation function to check against
     * @return this dataset, if compatible
     * @throws IllegalArgumentException if the dataset is not compatible
     *                                  with the given estimation function
     */
    public DataSet compatible(EstimationFunction estimationFunction) {
        if (!isCompatibleWith(estimationFunction)) {
            throw new IllegalArgumentException("Incompatible estimation function!");
        }
        return this;
    }
}
