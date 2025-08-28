package net.tvburger.jdl.common.patterns;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a type or member as part of the
 * <a href="https://en.wikipedia.org/wiki/Adapter_pattern">Adapter design pattern</a>.
 * <p>
 * The Adapter is a structural pattern that allows objects with
 * incompatible interfaces to collaborate by translating the interface
 * of one class into an interface expected by the client.
 * </p>
 *
 * <h2>Example:</h2>
 * <pre>{@code
 * // Target
 * interface JsonParser {
 *     Map<String, Object> parse(String json);
 * }
 *
 * // Adaptee (incompatible interface)
 * class LegacyParser {
 *     Document parseXml(String xml) { ... }
 * }
 *
 * // Adapter
 * @Adapter(role = Adapter.Role.ADAPTER)
 * class XmlToJsonAdapter implements JsonParser {
 *     private final LegacyParser adaptee = new LegacyParser();
 *     public Map<String, Object> parse(String json) {
 *         String xml = convertJsonToXml(json);
 *         return convertDocToMap(adaptee.parseXml(xml));
 *     }
 * }
 * }</pre>
 *
 * <p>
 * This annotation belongs to the
 * {@link DesignPattern.Category#STRUCTURAL} category and is retained in
 * source for documentation and static analysis.
 * </p>
 *
 * @see DesignPattern
 */
@DesignPattern(category = DesignPattern.Category.STRUCTURAL)
@Retention(RetentionPolicy.SOURCE)
@Target({ElementType.TYPE, ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface Adapter {

}