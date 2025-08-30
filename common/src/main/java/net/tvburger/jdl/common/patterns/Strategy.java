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

    enum Role {
        INTERFACE,
        CONCRETE
    }

    Role role();
}
