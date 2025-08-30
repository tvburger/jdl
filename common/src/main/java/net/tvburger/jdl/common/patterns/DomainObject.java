package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class as a <b>Domain Object</b>.
 *
 * <p>
 * A Domain Object represents a concept from the problem domain
 * (business, scientific, or technical domain) and is used to
 * model real-world entities inside the system.
 * </p>
 */
@Documented
@DesignPattern(DesignPattern.Category.DOMAIN_LANGUAGE)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface DomainObject {

}
