package net.tvburger.jdl.common.utils;

import net.tvburger.jdl.common.patterns.Holder;
import net.tvburger.jdl.common.patterns.StaticFactory;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * A simple generic holder class that encapsulates a value of type {@code K}.
 * <p>
 * This class is useful as a mutable reference wrapper, for example:
 * <ul>
 *   <li>Passing a value into lambdas or callbacks where reassignment is needed.</li>
 *   <li>Adjusting a stored value atomically using a function.</li>
 *   <li>Temporarily holding objects that may be cleared and reset later.</li>
 * </ul>
 *
 * @param <K> the type of the value being held
 */
@Holder
public class SimpleHolder<K> implements Supplier<K> {

    private K value;

    /**
     * Creates a new holder initialized to {@code null}.
     */
    @StaticFactory
    public SimpleHolder() {
        this(null);
    }

    /**
     * Creates a new holder with the given initial value.
     *
     * @param value the initial value to store, may be {@code null}
     */
    public SimpleHolder(K value) {
        this.value = value;
    }

    /**
     * Retrieves the current value stored in this holder.
     *
     * @return the current value, may be {@code null}
     */
    @Override
    public K get() {
        return value;
    }

    /**
     * Replaces the current value with the given one.
     *
     * @param value the new value to store, may be {@code null}
     */
    public void set(K value) {
        this.value = value;
    }

    /**
     * Adjusts the current value by applying the given function.
     * <p>
     * This method is synchronized to ensure that the adjustment
     * is applied atomically in multi-threaded environments.
     *
     * @param adjustment a function that takes the current value and returns the adjusted value
     * @throws NullPointerException if the adjustment function is {@code null}
     */
    public synchronized void adjust(Function<K, K> adjustment) {
        value = adjustment.apply(value);
    }


    /**
     * Clears the holder by setting its value to {@code null}.
     */
    public void clear() {
        value = null;
    }
}
