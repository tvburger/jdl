package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class or interface as part of the Strategy Pattern.
 * <p>
 * Typically applied to:
 * - The Strategy interface (defines the contract)
 * - Concrete Strategy implementations (different algorithms)
 */
@Documented
@DesignPattern(DesignPattern.Category.BEHAVIORAL)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Strategy {

    /**
     * Represents the role of a class or interface within the Strategy design pattern.
     *
     * <p>The Strategy pattern defines a family of algorithms, encapsulates each one,
     * and makes them interchangeable. Roles help clarify whether a type is the
     * abstract strategy interface or a concrete implementation of an algorithm.
     */
    enum Role {
        /**
         * Represents the abstract strategy interface in the pattern.
         */
        INTERFACE,
        /**
         * Represents a concrete implementation of the strategy.
         */
        CONCRETE
    }

    Role value();
}
