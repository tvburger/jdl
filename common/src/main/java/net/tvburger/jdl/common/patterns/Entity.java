package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class as an <b>Entity</b> in the domain model.
 *
 * <p>
 * An Entity represents a domain concept that has a distinct
 * <b>identity</b> which persists through state changes and across time.
 * Its equality is typically based on its identifier rather than its
 * attributes.
 * </p>
 *
 * <h3>Characteristics:</h3>
 * <ul>
 *   <li>Has a unique identifier (e.g., {@code id}).</li>
 *   <li>Identity remains constant, even as attributes change.</li>
 *   <li>Lifecycle is tracked (created, updated, deleted).</li>
 *   <li>Encapsulates domain state and rules relevant to its lifecycle.</li>
 * </ul>
 */
@DesignPattern(category = DesignPattern.Category.DOMAIN_LANGUAGE)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Entity {
}
