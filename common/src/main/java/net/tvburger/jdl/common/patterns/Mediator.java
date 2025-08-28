package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class or interface as part of the
 * <a href="https://en.wikipedia.org/wiki/Mediator_pattern">Mediator design pattern</a>.
 * <p>
 * The Mediator is a behavioral pattern that centralizes communication
 * between multiple collaborating objects ("colleagues"). Instead of
 * colleagues referencing each other directly, they interact only
 * through the mediator, which coordinates and routes their interactions.
 */
@DesignPattern(category = DesignPattern.Category.BEHAVIORAL)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Mediator {
}