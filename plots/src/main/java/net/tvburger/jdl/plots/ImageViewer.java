package net.tvburger.jdl.plots;

import net.tvburger.jdl.model.DataSet;

import javax.swing.*;
import java.awt.*;

public class ImageViewer extends JPanel implements DataRenderer {

    private final String label;
    private final Float[] pixels; // grayscale values (0.0 - 1.0)
    private final int width;
    private final int height;
    private final int scale; // zoom factor to make it visible
    private JFrame jframe;

    public static ImageViewer fromMnistImage(DataSet.Sample<Float> sample) {
        String label = "unknown";
        for (int i = 0; i < sample.targetCount(); i++) {
            if (sample.targetOutputs()[i] >= 0.9f) {
                label = Integer.toString(i);
                break;
            }
        }
        return new ImageViewer(label, sample.features(), 28, 28, 10);
    }

    public static ImageViewer fromPerceptronImage(String label, DataSet.Sample<Float> sample) {
        return new ImageViewer(label, sample.features(), 20, 20, 10);
    }

    public ImageViewer(String label, Float[] pixels, int width, int height, int scale) {
        if (pixels.length != width * height) {
            throw new IllegalArgumentException("Pixel array must have " + width + "x" + height + " = " + (width * height) + " elements.");
        }
        this.width = width;
        this.height = height;
        this.label = label;
        this.pixels = pixels;
        this.scale = scale;
        setPreferredSize(new Dimension(width * scale, height * scale));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float value = pixels[y * width + x];
                g.setColor(new Color(value, value, value)); // grayscale
                g.fillRect(x * scale, y * scale, scale, scale);
            }
        }
    }

    @Override
    public void display() {
        if (jframe != null) {
            return;
        }
        JFrame frame = new JFrame(label);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(this);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
