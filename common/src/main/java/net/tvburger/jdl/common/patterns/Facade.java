package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class or interface as a <b>Facade</b> in the
 * <a href="https://en.wikipedia.org/wiki/Facade_pattern">Facade design pattern</a>.
 * <p>
 * A Facade is a structural pattern that provides a simplified, high-level
 * interface to a complex subsystem. It shields clients from the details of
 * subsystem classes and reduces the learning curve by exposing only the
 * most commonly used functionality.
 * </p>
 *
 * <h2>Typical roles:</h2>
 * <ul>
 *   <li><b>Facade</b> – the class or interface annotated with {@code @Facade};
 *       it defines simple, coarse-grained methods for client use.</li>
 *   <li><b>Subsystem classes</b> – the detailed components hidden behind the facade;
 *       they perform the actual work.</li>
 *   <li><b>Client</b> – interacts only with the facade, not with subsystem classes directly.</li>
 * </ul>
 */
@DesignPattern(category = DesignPattern.Category.STRUCTURAL)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Facade {
}
