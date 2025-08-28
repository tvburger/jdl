package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class or type as a <b>Holder</b> structure.
 * <p>
 * A Holder is a lightweight helper pattern whose primary purpose is to
 * encapsulate and expose a reference to a value. It is often used in
 * situations where a simple mutable container is required, for example:
 * <ul>
 *   <li>Passing values into lambdas or anonymous classes where variables must be effectively final.</li>
 *   <li>Encapsulating a mutable reference for convenience or readability.</li>
 *   <li>Providing a simple mechanism for wrapping, clearing, or lazily updating a value.</li>
 * </ul>
 */
@DesignPattern(category = DesignPattern.Category.HELPER_STRUCTURES)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Holder {
}
