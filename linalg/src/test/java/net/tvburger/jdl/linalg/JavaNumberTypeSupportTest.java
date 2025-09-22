package net.tvburger.jdl.linalg;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.numbers.Rational;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class JavaNumberTypeSupportTest {

    @Test
    public void testRationalLong_valueOf() {
        // When
        Rational<Long> result = JavaNumberTypeSupport.RATIONAL_LONG.valueOf(0.11);

        // Then
        Assertions.assertEquals("11/100", result.toString());
    }

    @Test
    public void testRationalLong_valueOf_9() {
        // When
        Rational<Long> result = JavaNumberTypeSupport.RATIONAL_LONG.valueOf(9);

        // Then
        Assertions.assertEquals("9/1", result.toString());
    }

    @Test
    public void testRationalLong_valueOf_100() {
        // When
        Rational<Long> result = JavaNumberTypeSupport.RATIONAL_LONG.valueOf(100);

        // Then
        Assertions.assertEquals("100/1", result.toString());
    }

    @Test
    public void testRationalLong_valueOf_0_01() {
        // When
        Rational<Long> result = JavaNumberTypeSupport.RATIONAL_LONG.valueOf(0.01);

        // Then
        Assertions.assertEquals("1/100", result.toString());
    }
}
