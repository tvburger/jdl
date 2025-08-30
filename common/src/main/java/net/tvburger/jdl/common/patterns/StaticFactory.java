package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a class, method, or constructor as implementing the
 * <b>Static Factory</b> pattern.
 * <p>
 * The Static Factory pattern is an <i>object-graph</i> construction pattern
 * that provides an alternative to direct use of the implementation constructor
 * by exposing named factory methods/constructors. They are static and can be
 * used from any context.
 */
@Documented
@DesignPattern(DesignPattern.Category.OBJECT_GRAPH)
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE, ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface StaticFactory {
}
