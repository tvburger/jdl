package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class as participating in the <b>Decorator</b> design pattern.
 * <p>
 * The Decorator is a structural design pattern that allows behavior to be
 * added to individual objects, dynamically or statically, without modifying
 * their class. A decorator:
 * <ul>
 *   <li>Implements the same interface as the component it decorates.</li>
 *   <li>Wraps (composes) a reference to another component instance.</li>
 *   <li>Delegates operations to the wrapped component, while adding
 *       extra responsibilities before or after delegation.</li>
 * </ul>
 */
@DesignPattern(category = DesignPattern.Category.STRUCTURAL)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Decorator {
}