package net.tvburger.jdl.common.patterns;

import java.lang.annotation.*;

/**
 * Marker for design pattern annotations.
 * Acts as a top-level grouping.
 */
@Documented
@Retention(RetentionPolicy.SOURCE)
@Target(ElementType.ANNOTATION_TYPE)
public @interface DesignPattern {

    /**
     * Represents high-level categories of software design patterns.
     *
     * <p>Each category groups patterns with similar purposes and responsibilities.
     * These categories can help developers understand, organize, and communicate
     * design decisions more effectively.
     */
    enum Category {

        DOMAIN_LANGUAGE("Patterns that create a shared vocabulary and boundaries so requirements read like code and code reads like requirements.\n" +
                "Covers entities, value objects, aggregates, domain services, and other domain-driven design concepts."),
        STRUCTURAL("Organize objects into larger structures.\n" +
                "Covers Composite, Decorator, Adapter, Bridge, Facade, Proxy, Flyweight, composition principle."),
        OBJECT_GRAPH("Patterns that define how objects are created, located, wired, and assembled into working structures.\n" +
                "Covers object construction, factories, dependency injection, and service lookup."),
        EXTENSIBILITY("Patterns and architectural mechanisms that allow systems to be extended, customized, or configured without modifying existing code artifacts.\n" +
                "Covers Plug-in/Microkernel (module) architecture, Service Provider Interface (SPI), and Module/Package boundaries."),
        BEHAVIORAL("Patterns that define communication and control flow between objects, focusing on how responsibilities are distributed and interactions are structured.\n" +
                "Covers Strategy, Template Method, Observer, Command, State, Iterator, Visitor, Mediator, Memento, Chain of Responsibility, Interpreter."),
        HELPER_STRUCTURES("Lightweight idioms and helper patterns that provide convenience, readability, or reusability at the code level.\n" +
                "Often language-specific rather than architectural.");

        private final String explanation;

        /**
         * Constructs a Category with a detailed explanation.
         *
         * @param explanation a description of the category and the patterns it contains
         */
        Category(String explanation) {
            this.explanation = explanation;
        }

        /**
         * Returns the detailed explanation of this category.
         *
         * @return the explanation describing this category
         */
        public String getExplanation() {
            return explanation;
        }
    }

    Category value();

}