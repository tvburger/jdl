package net.tvburger.jdl.common.shapes;

public interface Shape3D extends Shape {

    default int getDimensions() {
        return 3;
    }

    int getWidth();

    int getHeight();

    int getDepth();

    static Shape3D of(Shape2D shape, int depth) {
        return of(shape.getWidth(), shape.getHeight(), depth);
    }

    static Shape3D of(int width, int height, int depth) {
        return new Shape3D() {
            @Override
            public int getWidth() {
                return width;
            }

            @Override
            public int getHeight() {
                return height;
            }

            @Override
            public int getDepth() {
                return depth;
            }
        };
    }
}
