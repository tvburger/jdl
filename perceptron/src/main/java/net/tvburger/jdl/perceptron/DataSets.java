package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.DataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;

public class DataSets {

    private DataSets() {
    }

    public static DataSet loadLinesAndCircles() {
        List<DataSet.Sample> samples = new ArrayList<>();
        try (Scanner scanner = new Scanner(Objects.requireNonNull(DataSets.class.getClassLoader().getResourceAsStream("lines-and-circles.csv")))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (!line.isEmpty()) {
                    String[] elements = line.split(",");
                    if (elements.length != 404) {
                        throw new IllegalArgumentException();
                    }
                    float circle = "circle".equals(elements[2]) ? 1.0f : 0.0f;
                    float left = "left".equals(elements[3]) ? 1.0f : 0.0f;
                    float[] targetOutputs = new float[]{circle, left, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    float[] features = new float[400];
                    for (int i = 0; i < 400; i++) {
                        features[i] = "0".equals(elements[i + 4]) ? 0.0f : 1.0f;
                    }
                    samples.add(new DataSet.Sample(features, targetOutputs));
                }
            }
        }
        return new DataSet(samples);
    }

    public static DataSet loadOr() {
        return loadLogicalDataSet("or");
    }

    public static DataSet loadAnd() {
        return loadLogicalDataSet("or");
    }

    public static DataSet loadXor() {
        return loadLogicalDataSet("and");
    }

    private static DataSet loadLogicalDataSet(String name) {
        List<DataSet.Sample> samples = new ArrayList<>();
        try (Scanner scanner = new Scanner(Objects.requireNonNull(DataSets.class.getClassLoader().getResourceAsStream(name + ".csv")))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (!line.isEmpty()) {
                    String[] elements = line.split(",");
                    if (elements.length != 4) {
                        throw new IllegalArgumentException();
                    }
                    float label = "1".equals(elements[1]) ? 1.0f : 0.0f;
                    float[] targetOutputs = new float[]{label};
                    float[] features = new float[2];
                    for (int i = 0; i < 2; i++) {
                        features[i] = "0".equals(elements[i + 2]) ? 0.0f : 1.0f;
                    }
                    samples.add(new DataSet.Sample(features, targetOutputs));
                }
            }
        }
        return new DataSet(samples);
    }
}
