package net.tvburger.jdl.datasets;

import net.tvburger.jdl.model.DataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;

public class LinesAndCircles implements DataSet.Loader<Float> {

    @Override
    public DataSet<Float> load() {
        List<DataSet.Sample<Float>> samples = new ArrayList<>();
        try (Scanner scanner = new Scanner(Objects.requireNonNull(LinesAndCircles.class.getClassLoader().getResourceAsStream("lines-and-circles.csv")))) {
            scanner.nextLine(); // skip header
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (!line.isEmpty()) {
                    String[] elements = line.split(",");
                    if (elements.length != 404) {
                        throw new IllegalArgumentException();
                    }
                    float circle = "circle".equals(elements[2]) ? 1.0f : 0.0f;
                    float left = "left".equals(elements[3]) ? 1.0f : 0.0f;
                    Float[] targetOutputs = new Float[]{circle, left, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    Float[] features = new Float[400];
                    for (int i = 0; i < 400; i++) {
                        features[i] = "0".equals(elements[i + 4]) ? 0.0f : 1.0f;
                    }
                    samples.add(new DataSet.Sample<>(features, targetOutputs));
                }
            }
        }
        return new DataSet<>(samples);
    }

}
