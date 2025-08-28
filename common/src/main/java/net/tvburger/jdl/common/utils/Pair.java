package net.tvburger.jdl.common.utils;

import net.tvburger.jdl.common.patterns.Composition;
import net.tvburger.jdl.common.patterns.StaticFactory;

/**
 * This class is a composition of 2 objects, named left and right.
 *
 * @param <L> the type of the left object
 * @param <R> the type of the right object
 */
@Composition
public class Pair<L, R> {

    private L left;
    private R right;
    boolean immutable;

    private Pair(L left, R right, boolean immutable) {
        this.left = left;
        this.right = right;
        this.immutable = immutable;
    }

    /**
     * Factory method to create a new immutable {@code Pair}.
     *
     * @param left  the left element
     * @param right the right element
     * @return a new {@code Pair} containing the given elements
     */
    @StaticFactory
    public static <L, R> Pair<L, R> of(L left, R right) {
        return immutable(left, right);
    }

    /**
     * Factory method to create a new mutable {@code Pair}.
     *
     * @param left  the left element
     * @param right the right element
     * @return a new {@code Pair} containing the given elements
     */
    @StaticFactory
    public static <L, R> Pair<L, R> mutable(L left, R right) {
        return new Pair<>(left, right, false);
    }

    /**
     * Factory method to create a new immutable {@code Pair}.
     *
     * @param left  the left element
     * @param right the right element
     * @return a new {@code Pair} containing the given elements
     */
    @StaticFactory
    public static <L, R> Pair<L, R> immutable(L left, R right) {
        return new Pair<>(left, right, true);
    }

    /**
     * Returns the left object
     *
     * @return the left object
     */
    public L left() {
        return left;
    }

    /**
     * Returns the right object
     *
     * @return the right object
     */
    public R right() {
        return right;
    }

    /**
     * Sets the left object
     *
     * @param left the left object to set
     * @throws UnsupportedOperationException if the pair is immutable
     */
    public void setLeft(L left) {
        if (immutable) {
            throw new UnsupportedOperationException();
        }
        this.left = left;
    }

    /**
     * Sets the right object
     *
     * @param right the right object to set
     * @throws UnsupportedOperationException if the pair is immutable
     */
    public void setRight(R right) {
        if (immutable) {
            throw new UnsupportedOperationException();
        }
        this.right = right;
    }
}
