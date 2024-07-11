package network;

import data.Image;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class Neuralnetwork {
    List<Layer> _layers = new ArrayList<>();
    double scale;

    public Neuralnetwork(List<Layer> _layers, double scale) {
        this._layers = _layers;
        this.scale = scale;
        linklayers();
    }


    //linking layers in list together
    private void linklayers() {


        if (_layers.size() <= 1) {
            return;
        }
        for (int i = 0; i < _layers.size(); i++) {
            if (i == 0) {
                _layers.get(i).set_nextLayer(_layers.get(i + 1));

            } else if (i == _layers.size() - 1) {
                _layers.get(i).set_prevLayer(_layers.get(i - 1));
            } else {
                _layers.get(i).set_nextLayer(_layers.get(i + 1));
                _layers.get(i).set_prevLayer(_layers.get(i - 1));
            }


        }


    }



// get errors
    public double[] getErr(double[] netoutput,int answer){
        int Classnum = netoutput.length;

        double[] expect = new  double[Classnum];
        expect[answer] = 1;


        // output - exppected
       return add(netoutput,multiply(expect,-1));


    }



    private int getMaxIndex(double[] in){

        double max = 0;
        int index = 0;

        for(int i = 0; i < in.length; i++){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }

        }

        return index;
    }

    public int guess(Image image){
        List<double[][]> inputList = new ArrayList<>();
        inputList.add(multiply(image.getData(), (1.0/scale)));

        double[] out = _layers.get(0).getOutput(inputList);
        int guess = getMaxIndex(out);

        return guess;
    }


    public float test (List<Image> images){
        int correct = 0;

        for(Image img: images){
            int guess = guess(img);

            if(guess == img.getLabel()){
                correct++;
            }
        }

        return((float)correct/images.size());
    }

    public void train (List<Image> images){

        for(Image img:images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(img.getData(), (1.0/scale)));

            double[] out = _layers.get(0).getOutput(inList);
            double[] dldO = getErr(out, img.getLabel());

            _layers.get((_layers.size()-1)).backPropagation(dldO);
        }
}
    }

