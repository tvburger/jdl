package net.tvburger.jdl.common.numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.math.BigInteger;

public class RationalTest {

    @Test
    public void testRational() {
        // Given
        BigInteger numerator = new BigInteger("48281997035330229036818955958401861443229634088684181635679091522496730162146053985835418957769160012584807130471556740729003310800741758482894832416146077113052305270137876407424805361937951872056633107878293978652901808095172003671651640950162980871709750574799473082894284880927885613728649565342761081414255637");
        BigInteger denominator = new BigInteger("1627699625506545813956948468608602467719550028266661209860157740780804812365581770580308719015368571107627299753324394763851072034030087863020053840332826077221777330553352309443239663832107167269963961266714145978807434724568468795929435288570839510223419839749739131089627164928803693972192362665492546875000000000");

        // When
        Rational<BigInteger> result = new Rational<>(numerator, denominator);

        // Then
        Assertions.assertNotEquals(Double.NaN, result.numerator().doubleValue());
        Assertions.assertEquals(Double.POSITIVE_INFINITY, result.numerator().doubleValue());
        Assertions.assertNotEquals(Double.NaN, result.denominator().doubleValue());
        Assertions.assertEquals(Double.POSITIVE_INFINITY, result.numerator().doubleValue());
        Assertions.assertEquals(Double.NaN, result.doubleValue());
    }
}
