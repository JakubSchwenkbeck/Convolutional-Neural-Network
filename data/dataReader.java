package data;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class dataReader {
    //class to read mnist data from files into arrays
    private static final int rows = 28;
    private static final int cols = 28;


    public static List<Image> readData(String path)  {

    List<Image> Images = new ArrayList<>();

    // try read the file
    try(BufferedReader datareader = new BufferedReader(new FileReader(path))){

        String line;
        //while not in the end of the file
        while((line = datareader.readLine() )  != null){
            String[] linechars = line.split(",");// split into segments(chars kinda)

            double data[][] = new double[rows][cols]; //data array : Representation as Matrix
            int label = Integer.parseInt(linechars[0]); //first entry is the label

            int index = 1;
            for(int row = 0; row < rows ; row++) {
                for (int col = 0; col < cols; col++) {
                    data[row][col] = (double) Integer.parseInt(linechars[index]); //fill matrix
                    index++;


                }
            }

        Images.add(new Image(data, label)); //add to list

        }

    }catch(Exception e){
        //catch really nothing tbh
        System.out.println("Oh no there is an error with reading the file! Please check path and try again");
        }

        return Images;
    }


}
