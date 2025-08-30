package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class or method as participating in the
 * <b>Factory Method</b> design pattern.
 * <p>
 * The Factory Method is a creational pattern where a method
 * defines the interface for creating an object, but lets
 * subclasses or implementations decide which concrete type
 * to instantiate. This allows a class to delegate object
 * creation to subclasses without being tightly coupled to
 * specific implementations.
 */
@Documented
@DesignPattern(DesignPattern.Category.OBJECT_GRAPH)
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface FactoryMethod {
}
