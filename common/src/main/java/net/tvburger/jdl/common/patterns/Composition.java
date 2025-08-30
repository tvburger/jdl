package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class or type as using the <b>Composition</b> design pattern.
 * <p>
 * Composition is a structural design pattern that builds complex objects
 * by combining or "composing" simpler objects.
 */
@Documented
@DesignPattern(DesignPattern.Category.STRUCTURAL)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Composition {
}
