package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a class or type as using the <b>Composition</b> design pattern.
 * <p>
 * Composition is a structural design pattern that builds complex objects
 * by combining or "composing" simpler objects.
 */
@DesignPattern(category = DesignPattern.Category.STRUCTURAL)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Composition {
}
