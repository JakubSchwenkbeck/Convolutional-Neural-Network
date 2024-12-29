package network;

import java.io.*;

public class NetworkUtils {

    public static void saveNetwork(NeuralNetwork network, String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(network);
            System.out.println("Network saved to " + filePath);
        } catch (IOException e) {
            System.err.println("Error saving network: " + e.getMessage());
        }
    }

    public static NeuralNetwork loadNetwork(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            NeuralNetwork network = (NeuralNetwork) ois.readObject();
            System.out.println("Network loaded from " + filePath);
            return network;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading network: " + e.getMessage());
            return null;
        }
    }
}
