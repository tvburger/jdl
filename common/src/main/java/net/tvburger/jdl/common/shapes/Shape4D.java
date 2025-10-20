package net.tvburger.jdl.common.shapes;

public interface Shape4D extends Shape {

    default int getDimensions() {
        return 4;
    }

    int getWidth();

    int getHeight();

    int getDepth();

    int getCount();

    static Shape4D of(Shape3D shape, int count) {
        return of(shape.getWidth(), shape.getHeight(), shape.getDepth(), count);
    }

    static Shape4D of(int width, int height, int depth, int count) {
        return new Shape4D() {
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

            @Override
            public int getCount() {
                return count;
            }
        };
    }
}
