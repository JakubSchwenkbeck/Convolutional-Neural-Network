package training;

import data.Image;
import data.dataReader;
import network.NetworkBuilder;
import network.NetworkUtils;
import network.NeuralNetwork;

import java.util.List;

import static java.util.Collections.shuffle;

public class TrainNetwork {

    public static void train(){
        long SEED = 123;

        System.out.println("Starting data loading...");
        String networkFile = "saved_network.dat";

        List<Image> imagesTest = dataReader.readData("Data/mnist_test.csv");
        List<Image> imagesTrain = dataReader.readData("Data/mnist_train.csv");

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28,28,256*100);
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3,2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();
        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        int epochs = 5;

        for(int i = 0; i < epochs; i++){
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
            NetworkUtils.saveNetwork(net, networkFile); // Save after each epoch


        }
    }
}
