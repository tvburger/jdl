package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a type as an implementation of the <em>Builder</em> design pattern.
 *
 * <p>The Builder pattern is a <strong>creational design pattern</strong> that separates
 * the construction of a complex object from its representation. It provides a fluent
 * or step-by-step API for assembling the parts of an object before producing the final,
 * immutable result.</p>
 *
 * <h3>Intent</h3>
 * <ul>
 *   <li>Encapsulate complex construction logic.</li>
 *   <li>Support optional parameters without telescoping constructors.</li>
 *   <li>Provide a fluent interface for readability.</li>
 *   <li>Allow reuse of the same builder to create multiple instances with variations.</li>
 * </ul>
 */
@Documented
@DesignPattern(DesignPattern.Category.OBJECT_GRAPH)
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.TYPE)
public @interface Builder {

}