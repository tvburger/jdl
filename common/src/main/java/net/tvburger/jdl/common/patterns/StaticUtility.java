package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class as a Static Utility class:
 * - Only static methods
 * - No state (no instance fields)
 * - Not meant to be instantiated
 */
@Documented
@DesignPattern(DesignPattern.Category.HELPER_STRUCTURES)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface StaticUtility {
}
