
import data.Image;
import data.dataReader;
import network.NetworkBuilder;

import network.Neuralnetwork;

import java.util.List;

import static java.util.Collections.shuffle;

public class Main {

    public static void main(String[] args) {

        long SEED = 123;

        System.out.println("Starting data loading...");

        List<Image> imagesTest = new dataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new dataReader().readData("data/mnist_train.csv");

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28,28,700);
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3,2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        Neuralnetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        int epochs = 3;

        for(int i = 0; i < epochs; i++){
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
        }
    }
}