package net.tvburger.jdl.common.shapes;

public interface Shape1D extends Shape {

    default int getDimensions() {
        return 1;
    }

    int getLength();

    static Shape1D of(int length) {
        return new Shape1D() {
            @Override
            public int getLength() {
                return length;
            }
        };
    }

}
