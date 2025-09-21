package net.tvburger.jdl.plots;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class Table implements DataRenderer {

    private final String title;
    private final List<Object[]> data = new ArrayList<>();
    private JTable table;
    private String[] columnNames;

    public Table(String title) {
        this.title = title;
    }

    public void addEntry(Object[] values) {
        data.add(values);
    }

    public void setColumnNames(String... columnNames) {
        this.columnNames = columnNames;
    }

    @Override
    public void display() {
        Object[][] values = new Object[data.size()][];
        int i = 0;
        for (Object[] row : data) {
            values[i++] = row;
        }
        table = new JTable(values, columnNames);
        JScrollPane tableScroll = new JScrollPane(table);

        JFrame frame = new JFrame(title);
        frame.setLayout(new BorderLayout());
        frame.add(table, BorderLayout.CENTER);
        frame.add(tableScroll, BorderLayout.EAST);

        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}
