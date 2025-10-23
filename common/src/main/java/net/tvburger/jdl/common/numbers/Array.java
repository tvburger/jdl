package net.tvburger.jdl.common.numbers;

import net.tvburger.jdl.common.function.UnaryFunction;

import java.util.Iterator;
import java.util.Objects;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public interface Array<N> extends Tensor<N>, Iterable<N>, Cloneable {

    @Override
    default int[] dimensions() {
        return new int[length()];
    }

    @Override
    default N get(int... i) {
        return get(i[0]);
    }

    N get(int index);

    void set(int index, N value);

    default void set(Array<N> values) {
        set(values, 0, values.length());
    }

    default void set(Array<N> values, int offset) {
        set(values, offset, values.length() - offset);
    }

    void set(Array<N> values, int offset, int length);

    int length();

    default Array<N> slice(int offset) {
        return slice(offset, length() - offset, 1);
    }

    default Array<N> slice(int offset, int length) {
        return slice(offset, length, 1);
    }

    Array<N> slice(int offset, int length, int stride);

    Array<N> apply(UnaryFunction<N, N> function);

    N[] backingArray();

    int offset();

    int stride();

    Array<N> clone();

    class Impl<N> implements Array<N>, Cloneable {

        private final N[] values;
        private final int offset;
        private final int length;
        private final int stride;

        public Impl(N[] values, int offset, int length, int stride) {
            this.values = values;
            this.offset = offset;
            this.length = length;
            this.stride = stride;
        }

        @Override
        public N get(int index) {
            return values[offset + index * stride];
        }

        @Override
        public void set(int index, N value) {
            values[offset + index * stride] = value;
        }

        @Override
        public void set(Array<N> values, int offset, int length) {
            for (int i = 0, j = offset; i < length; i++, j++) {
                this.values[this.offset + j * this.stride] = values.get(i);
            }
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public Array<N> slice(int offset, int length, int stride) {
            return new Impl<>(values, this.offset + offset * this.stride, length * this.stride, this.stride * stride);
        }

        @Override
        public Array<N> apply(UnaryFunction<N, N> function) {
            int limit = offset + length * stride;
            for (int i = offset; i < limit; i += stride) {
                values[i] = function.mapToScalar(values[i]);
            }
            return this;
        }

        @Override
        public N[] backingArray() {
            return values;
        }

        @Override
        public int offset() {
            return offset;
        }

        @Override
        public int stride() {
            return stride;
        }

        @Override
        public Impl<N> clone() {
            return new Impl<>(values.clone(), offset, length, stride);
        }

        @Override
        public Iterator<N> iterator() {
            return new Iterator<>() {

                private final int limit = offset + length * stride;
                private int nextIndex = offset;

                @Override
                public boolean hasNext() {
                    return nextIndex < limit;
                }

                @Override
                public N next() {
                    N value = values[nextIndex];
                    nextIndex += stride;
                    return value;
                }
            };
        }

        @Override
        public String toString() {
            return Array.toString(this);
        }
    }

    static <N> Array<N> of(N[] values) {
        return of(values, 0, values.length);
    }

    static <N> Array<N> of(N[] values, int offset) {
        return of(values, offset, values.length);
    }

    static <N> Array<N> of(N[] values, int offset, int length) {
        return new Array<>() {

            @Override
            public Iterator<N> iterator() {
                return new Iterator<>() {

                    private final int limit = offset + length;
                    private int nextIndex = offset;

                    @Override
                    public boolean hasNext() {
                        return nextIndex < limit;
                    }

                    @Override
                    public N next() {
                        N value = values[nextIndex];
                        nextIndex++;
                        return value;
                    }
                };
            }

            @Override
            public N get(int index) {
                return values[offset + index];
            }

            @Override
            public void set(int index, N value) {
                values[offset + index] = value;
            }

            @Override
            public void set(Array<N> _values, int _offset, int _length) {
                if (_values.stride() == 1) {
                    System.arraycopy(_values.backingArray(), _values.offset(), values, offset + _offset, _length);
                } else {
                    for (int i = 0, j = offset; i < length; i++, j++) {
                        values[j] = _values.get(i);
                    }
                }
            }

            @Override
            public int length() {
                return length;
            }

            @Override
            public Array<N> slice(int offset, int length, int stride) {
                return new Impl<>(values, offset, length, stride);
            }

            @Override
            public Array<N> apply(UnaryFunction<N, N> function) {
                int limit = offset + length;
                for (int i = offset; i < limit; i++) {
                    values[i] = function.mapToScalar(values[i]);
                }
                return this;
            }

            @Override
            public N[] backingArray() {
                return values;
            }

            @Override
            public int offset() {
                return offset;
            }

            @Override
            public int stride() {
                return 1;
            }

            @Override
            public Array<N> clone() {
                return of(values.clone());
            }

            @Override
            public String toString() {
                return Array.toString(this);
            }
        };
    }

    static <N extends Number> Array<N> of(N[] values, int from, int length, int stride) {
        return new Impl<>(values, from, length, stride);
    }

    static <N extends Number> int hashCode(Array<N> array) {
        int result = 0;
        for (N element : array) {
            result += Objects.hashCode(element);
        }
        return result;
    }

    static <N extends Number> boolean equals(Array<N> array, Array<N> other, JavaNumberTypeSupport<N> typeSupport) {
        if (array.length() != other.length()) {
            return false;
        }
        Iterator<N> thisIterator = other.iterator();
        Iterator<N> otherIterator = array.iterator();
        while (thisIterator.hasNext() && otherIterator.hasNext()) {
            N thisValue = thisIterator.next();
            N otherValue = otherIterator.next();
            if (typeSupport.compare(thisValue, otherValue) != 0) {
                return false;
            }
        }
        return true;
    }

    static boolean equals(Array<?> array, Array<?> other) {
        if (array.length() != other.length()) {
            return false;
        }
        Iterator<?> thisIterator = other.iterator();
        Iterator<?> otherIterator = array.iterator();
        while (thisIterator.hasNext() && otherIterator.hasNext()) {
            Object thisValue = thisIterator.next();
            Object otherValue = otherIterator.next();
            if (!thisValue.equals(otherValue)) {
                return false;
            }
        }
        return true;
    }

    static String toString(Array<?> array) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(array.get(i));
        }
        return sb.append("]@").append(Integer.toHexString(System.identityHashCode(array))).toString();
    }

    Array<?> EMPTY = empty();

    @SuppressWarnings("unchecked")
    static <N> Array<N> empty() {
        return (Array<N>) EMPTY;
    }

    static <N> Stream<N> stream(Array<N> array) {
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(array.iterator(), 0),
                false
        );
    }
}
