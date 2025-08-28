package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class as a <b>Value Object</b> in the domain model.
 *
 * <p>
 * A Value Object represents a domain concept that is defined
 * entirely by its <b>attributes</b>, with no identity of its own.
 * Value Objects equality is based on comparing their values.
 * </p>
 *
 * <h3>Characteristics:</h3>
 * <ul>
 *   <li>No unique identifier; identity is not tracked.</li>
 *   <li>Equality is by comparing all attributes.</li>
 * </ul>
 */
@DesignPattern(category = DesignPattern.Category.HELPER_STRUCTURES)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface ValueObject {
}
