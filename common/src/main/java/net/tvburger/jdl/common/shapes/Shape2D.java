package net.tvburger.jdl.common.shapes;

public interface Shape2D extends Shape {

    default int getDimensions() {
        return 2;
    }

    int getWidth();

    int getHeight();

    static Shape2D of(Shape1D shape, int height) {
        return of(shape.getLength(), height);
    }

    static Shape2D of(int width, int height) {
        return new Shape2D() {
            @Override
            public int getWidth() {
                return width;
            }

            @Override
            public int getHeight() {
                return height;
            }
        };
    }
}
