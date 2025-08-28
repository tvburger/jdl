package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class as a <b>Domain Object</b>.
 *
 * <p>
 * A Domain Object represents a concept from the problem domain
 * (business, scientific, or technical domain) and is used to
 * model real-world entities inside the system.
 * </p>
 */
@DesignPattern(category = DesignPattern.Category.DOMAIN_LANGUAGE)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface DomainObject {

}
