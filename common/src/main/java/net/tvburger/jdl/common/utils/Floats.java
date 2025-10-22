package net.tvburger.jdl.common.utils;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.patterns.StaticUtility;

/**
 * Implements utility functions to operate on floats
 */
@StaticUtility
public final class Floats {

    /**
     * The EPSILON we are using for considering floats equal. (solve problems like: 0,99999999... vs 1,0)
     */
    public static final float EPSILON = 1e-6f;

    public static final float[] EMPTY = new float[0];

    private Floats() {
    }

    /**
     * Returns true if float a and b are closer than EPSILON from each other.
     *
     * @param a the float to compare to b
     * @param b the float to compare to a
     * @return if the floats are (enough) equal
     */
    public static boolean equals(float a, float b) {
        return Math.abs(a - b) < EPSILON;
    }

    /**
     * Returns true if float a and b are not equal (differ)
     *
     * @param a the float to compare to b
     * @param b the float to compare to a
     * @return true if the floast are different
     * @see #equals(float, float)
     */
    public static boolean notEquals(float a, float b) {
        return !equals(a, b);
    }

    /**
     * Easy way to create float arrays in java
     *
     * @param f the floats to put in the array
     * @return the array
     */
    public static Array<Float> of(Float... f) {
        return Array.of(f);
    }

    /**
     * Returns true if f1 is greater than f2
     *
     * @param f1 the float to compare to f2
     * @param f2 the float to compare to f1
     * @return if f1 greater than f2
     * @see #notEquals(float, float)
     */
    public static boolean greaterThan(float f1, float f2) {
        return f1 > f2 && notEquals(f1, f2);
    }

    /**
     * Converts an array of floats to a boolean array. True if the float is positive
     *
     * @param floats the floats to convert
     * @return the corresponding booleans
     */
    public static Array<Boolean> toBooleans(Array<Float> floats, float threshold) {
        Array<Boolean> booleans = Array.of(new Boolean[floats.length()]);
        for (int i = 0; i < floats.length(); i++) {
            booleans.set(i, floats.get(i) >= threshold);
        }
        return booleans;
    }

    public static boolean[] toBooleans(Float[] floats, float threshold) {
        boolean[] booleans = new boolean[floats.length];
        for (int i = 0; i < floats.length; i++) {
            booleans[i] = floats[i] >= threshold;
        }
        return booleans;
    }
}
