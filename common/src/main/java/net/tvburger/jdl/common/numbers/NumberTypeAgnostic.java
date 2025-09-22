package net.tvburger.jdl.common.numbers;

public interface NumberTypeAgnostic<N extends Number> {

    JavaNumberTypeSupport<N> getCurrentNumberType();

}
