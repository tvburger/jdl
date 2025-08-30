package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marks a type or method as an implementation of the <em>Proxy</em> design pattern.
 *
 * <p>The Proxy pattern is a <strong>structural design pattern</strong> that provides
 * a surrogate or placeholder for another object to control access to it. Proxies
 * typically forward calls to a real subject while adding extra behavior such as
 * access control, lazy initialization, monitoring, or indirection.</p>
 *
 * <h3>Typical proxy roles</h3>
 * <ul>
 *   <li><strong>Virtual proxy:</strong> defers expensive object creation until it is needed.</li>
 *   <li><strong>Protection proxy:</strong> controls access rights to the underlying object.</li>
 *   <li><strong>Remote proxy:</strong> represents an object in a different address space or JVM.</li>
 *   <li><strong>Caching proxy:</strong> adds caching to expensive operations.</li>
 *   <li><strong>Monitoring proxy:</strong> collects statistics or logging information.</li>
 * </ul>
 */
@Documented
@DesignPattern(DesignPattern.Category.STRUCTURAL)
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface Proxy {

}