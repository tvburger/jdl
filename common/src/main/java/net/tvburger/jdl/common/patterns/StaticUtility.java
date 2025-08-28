package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class as a Static Utility class:
 * - Only static methods
 * - No state (no instance fields)
 * - Not meant to be instantiated
 */
@DesignPattern(category = DesignPattern.Category.HELPER_STRUCTURES)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface StaticUtility {
}
