package net.tvburger.jdl.datasets;

import java.io.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class MnistReader {

    public static final int MAGIC_LABELS = 2049; // >II
    public static final int MAGIC_IMAGES = 2051; // >IIII

    public static class MnistData {
        public final List<int[]> images; // each length = rows*cols
        public final byte[] labels;      // unsigned values 0..9
        public final int rows;
        public final int cols;

        public MnistData(List<int[]> images, byte[] labels, int rows, int cols) {
            this.images = images;
            this.labels = labels;
            this.rows = rows;
            this.cols = cols;
        }
    }

    public static MnistData readImagesLabels(Path imagesPath, Path labelsPath) throws IOException {
        // ---- Read labels ----
        final int labelCount;
        final byte[] labels;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsPath.toFile())))) {
            int magic = in.readInt(); // big-endian by default
            if (magic != MAGIC_LABELS) {
                throw new IOException("Magic number mismatch for labels: expected " + MAGIC_LABELS + ", got " + magic);
            }
            labelCount = in.readInt();
            labels = in.readNBytes(labelCount);
            if (labels.length != labelCount) {
                throw new EOFException("Labels file truncated: expected " + labelCount + " bytes, got " + labels.length);
            }
        }

        // ---- Read images ----
        final int imageCount, rows, cols;
        final byte[] imageBytes;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesPath.toFile())))) {
            int magic = in.readInt();
            if (magic != MAGIC_IMAGES) {
                throw new IOException("Magic number mismatch for images: expected " + MAGIC_IMAGES + ", got " + magic);
            }
            imageCount = in.readInt();
            rows = in.readInt();
            cols = in.readInt();
            long total = (long) imageCount * rows * cols;
            if (total > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Image data too large: " + total + " bytes");
            }
            imageBytes = in.readNBytes((int) total);
            if (imageBytes.length != total) {
                throw new EOFException("Images file truncated: expected " + total + " bytes, got " + imageBytes.length);
            }
        }

        if (imageCount != labelCount) {
            throw new IOException("Count mismatch: images=" + imageCount + ", labels=" + labelCount);
        }

        // ---- Convert to int[] per image (0..255) ----
        List<int[]> images = new ArrayList<>(imageCount);
        final int pixelsPerImage = rows * cols;
        for (int i = 0; i < imageCount; i++) {
            int[] img = new int[pixelsPerImage];
            int base = i * pixelsPerImage;
            for (int p = 0; p < pixelsPerImage; p++) {
                // Convert signed byte to unsigned int 0..255
                img[p] = imageBytes[base + p] & 0xFF;
            }
            images.add(img);
        }

        return new MnistData(images, labels, rows, cols);
    }
}