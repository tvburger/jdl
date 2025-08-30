package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Indicates that the annotated type or member is a {@code Marker Interface} pattern.
 * <p>
 * A <em>marker interface</em> (also known as a tagging interface) is an interface
 * with no methods, used to "tag" or "mark" a class or component as having a
 * particular semantic role. The presence of the marker is then detected by
 * frameworks, tools, or reflection logic to enable special behavior without
 * modifying the existing code.
 * </p>
 *
 * <p>Examples in the Java standard library include:
 * <ul>
 *   <li>{@link java.io.Serializable} — marks a class as capable of being serialized</li>
 *   <li>{@link java.lang.Cloneable} — marks a class as supporting {@code Object.clone()}</li>
 *   <li>{@link java.rmi.Remote} — marks a class as usable for remote method invocation</li>
 * </ul>
 * </p>
 *
 * <p>
 * This annotation belongs to the {@link DesignPattern.Category#EXTENSIBILITY EXTENSIBILITY}
 * category, since marker interfaces provide a lightweight mechanism for systems to
 * be extended, customized, or configured without altering the code of the marked class.
 * </p>
 *
 * <p>Intended usage:
 * <ul>
 *   <li>On an interface that is designed to serve as a marker interface</li>
 *   <li>Optionally on methods or constructors when a framework uses markers on call sites</li>
 * </ul>
 * </p>
 *
 * @see java.io.Serializable
 * @see java.lang.Cloneable
 * @see java.rmi.Remote
 */
@Documented
@DesignPattern(DesignPattern.Category.EXTENSIBILITY)
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE, ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface MarkerInterface {

}