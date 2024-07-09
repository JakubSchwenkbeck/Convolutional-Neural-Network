package data;

public class Image {
    // Class describing the data this CNN operates on:
    // Images are in the mnist format (28x28 pixels w/ grey value)
    // An Image has a label (int from 0 - 9) and each pixel is represented as a double grey value

    private  int label;

    private  double[][] data;


  //constructor
    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    //getter
    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }


    //custom to String function to display in terminal (for easy testing)

@Override
   public String toString(){

        String s = label +" ,\n";
    for (int i = 0; i < data.length; i++) {
        for (int j = 0; j < data[0].length; j++) {
            s+= data[i][j] + ", ";
        }
        s+= "\n";
    }
return s;
   }

}
