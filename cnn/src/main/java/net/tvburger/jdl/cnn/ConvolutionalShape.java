package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.shapes.Shape4D;

public class ConvolutionalShape implements Shape4D, Cloneable {

    public static final int DEFAULT_CHANNELS = 1;
    public static final int DEFAULT_CHANNEL = 1;
    public static final int DEFAULT_COUNT = 1;
    public static final int DEFAULT_INDEX = 0;

    private final int width;
    private final int height;
    private final int channels;
    private final int count;
    private final Array<Float> elements;

    public ConvolutionalShape(int width, int height) {
        this(width, height, DEFAULT_CHANNELS);
    }

    public ConvolutionalShape(int width, int height, int channels) {
        this(width, height, channels, DEFAULT_COUNT);
    }

    public ConvolutionalShape(int width, int height, int channels, int count) {
        this(width, height, channels, count, JavaNumberTypeSupport.FLOAT.createArray(width * height * channels * count));
    }

    public ConvolutionalShape(int width, int height, int channels, int count, Array<Float> elements) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.count = count;
        this.elements = elements;
    }

    public Array<Float> getElements() {
        return elements;
    }

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
        return channels;
    }

    @Override
    public int getCount() {
        return count;
    }

    public void setElement(float value, int x, int y) {
        setElement(value, x, y, DEFAULT_CHANNEL);
    }

    public void setElement(float value, int x, int y, int channel) {
        setElement(value, x, y, channel, DEFAULT_INDEX);
    }

    public void setElement(float value, int x, int y, int channel, int index) {
        elements.set(index * (width * height * channels) + (channel - 1) * (width * height) + y * width + x, value);
    }

    public float getElement(int x, int y) {
        return getElement(x, y, DEFAULT_CHANNEL);
    }

    public float getElement(int x, int y, int channel) {
        return getElement(x, y, channel, DEFAULT_INDEX);
    }

    public float getElement(int x, int y, int channel, int index) {
        return elements.get(index * (width * height * channels) + (channel - 1) * (width * height) + y * width + x);
    }

    public int getChannels() {
        return channels;
    }

    @Override
    public ConvolutionalShape clone() {
        return new ConvolutionalShape(width, height, channels, count, elements == null ? JavaNumberTypeSupport.FLOAT.createArray(width * height * channels * count) : elements.clone());
    }

    public ConvolutionalShape withElements(Array<Float> elements) {
        return new ConvolutionalShape(width, height, channels, count, elements);
    }
}
