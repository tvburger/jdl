package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class or method as participating in the
 * <b>Template Method</b> design pattern.
 * <p>
 * The Template Method is a behavioral design pattern that defines the
 * <i>skeleton of an algorithm</i> in a base class (the template) while
 * allowing subclasses to redefine specific steps without changing the
 * overall algorithm structure.
 */
@Documented
@DesignPattern(DesignPattern.Category.EXTENSIBILITY)
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface TemplateMethod {

}
