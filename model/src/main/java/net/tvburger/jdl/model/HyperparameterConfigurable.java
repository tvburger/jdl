package net.tvburger.jdl.model;

import java.util.Map;

/**
 * Interface for components that expose tunable hyperparameters.
 * <p>
 * Hyperparameters are configuration values that control the behavior of a model
 * or algorithm (e.g., learning rate, momentum, batch size) but are not learned
 * directly from data. This interface provides a generic mechanism to inspect
 * and update such values at runtime.
 */
public interface HyperparameterConfigurable {

    default boolean hasHyperparameter(String name) {
        return getHyperparameters().containsKey(name);
    }

    /**
     * Returns a snapshot of all current hyperparameters.
     * <p>
     * The returned map should contain stable, human-readable keys (such as
     * {@code "learningRate"} or {@code "momentum"}) mapped to their current values.
     * Implementations are encouraged to return a defensive copy so that external
     * modifications do not affect the internal state.
     *
     * @return an immutable or defensive copy of the current hyperparameters
     */
    Map<String, Object> getHyperparameters();

    /**
     * Returns the value of a single hyperparameter by name.
     * <p>
     * This is a convenience method that queries {@link #getHyperparameters()}.
     * If the name is not present, this will return {@code null}.
     *
     * @param name the hyperparameter name
     * @return the hyperparameter value, or {@code null} if not defined
     */
    default Object getHyperparameter(String name) {
        return getHyperparameters().get(name);
    }

    /**
     * Returns the value of a single hyperparameter by name, cast to the expected type.
     * <p>
     * This is a convenience method that queries {@link #getHyperparameters()} and
     * casts the result to the requested type.
     * If the name is not present, this will return {@code null}.
     * If the value is present but not of the expected type, a {@link ClassCastException}
     * will be thrown.
     *
     * @param name                    the hyperparameter name
     * @param hyperparameterTypeClass the expected type of the value
     * @param <T>                     the type parameter corresponding to the expected value type
     * @return the hyperparameter value cast to the requested type, or {@code null} if not defined
     * @throws ClassCastException if the value cannot be cast to the given type
     */
    default <T> T getHyperparameter(String name, Class<T> hyperparameterTypeClass) {
        return hyperparameterTypeClass.cast(getHyperparameters().get(name));
    }

    /**
     * Updates multiple hyperparameters at once by name.
     * <p>
     * Implementations should validate the input and either:
     * <ul>
     *   <li>ignore unknown keys, or</li>
     *   <li>throw an {@link IllegalArgumentException} if an unrecognized
     *       hyperparameter is provided.</li>
     * </ul>
     *
     * @param hyperparameters a map of hyperparameter names to new values
     * @throws IllegalArgumentException if any value is invalid or a key is unrecognized
     */
    default void setHyperparameters(Map<String, Object> hyperparameters) {
        hyperparameters.forEach(this::setHyperparameter);
    }

    /**
     * Updates a single hyperparameter by name.
     *
     * @param name  the hyperparameter name
     * @param value the new hyperparameter value
     * @throws IllegalArgumentException if the name is unrecognized or the value is invalid
     */
    void setHyperparameter(String name, Object value);

}
