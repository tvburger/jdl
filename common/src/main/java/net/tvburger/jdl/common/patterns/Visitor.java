package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Represents the <a href="https://en.wikipedia.org/wiki/Visitor_pattern">Visitor design pattern</a>.
 *
 * <p>A Visitor is a behavioral design pattern that separates an algorithm from the
 * objects it operates on. Instead of embedding behavior in the elements themselves,
 * a Visitor class encapsulates operations and “visits” elements to perform them.
 *
 * <p>This allows you to:
 * <ul>
 *     <li>Add new operations without modifying existing element classes</li>
 *     <li>Keep related operations together in one class</li>
 *     <li>Traverse complex object structures consistently</li>
 * </ul>
 */
@Documented
@DesignPattern(DesignPattern.Category.BEHAVIORAL)
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Visitor {
}
