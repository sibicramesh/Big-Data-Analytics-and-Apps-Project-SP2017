import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.TreeMap;

/**
 * Created by Barath on 02-08-2017.
 */
public class ImageAnnotation {
    public static void main(String[] args) throws IOException {
        final ClarifaiClient client = new ClarifaiBuilder("fLiPpuBuVY9S2agDztOQc8dBYoSyguXR0GxgicKy", "OEQpx7HzQHa2cLdr_hfhobyo7OWy61i85a_qhpyT")
                .client(new OkHttpClient()) // OPTIONAL. Allows customization of OkHttp by the user
                .buildSync(); // or use .build() to get a Future<ClarifaiClient>
        client.getToken();

        File file = new File("output/mainframes");
        File[] files = file.listFiles();
        TreeMap tmap_score = new TreeMap(Collections.reverseOrder());
        TreeMap tmap_Label = new TreeMap();
        int labelCount=0;
        System.out.println("***************************************************************");
        System.out.println("********************** Video Summary **************************");
        System.out.println("***************************************************************");
        for (int i=0; i<files.length;i++){
            //Getting the response from Clarifai API
            ClarifaiResponse response = client.getDefaultModels().generalModel().predict()
                    .withInputs(
                            ClarifaiInput.forImage(ClarifaiImage.of(files[i]))
                    ).executeSync();
            List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response.get();
            MBFImage image = ImageUtilities.readMBF(files[i]);
            int x = image.getWidth();
            int y = image.getHeight();

            //Get only response data with name and confidence score into List
            List<Concept> data = predictions.get(0).data();

            System.out.println("*************" + files[i] + "***********");
                        for (int j = 0; j < data.size(); j++) {
                System.out.println(data.get(j).name() + " - " + data.get(j).value());
            }
            /*To find top unique score in whole video
             *Treemap data structure eliminate duplicate and its faster to put Key value pair with constant time complexity*/
            for (int j = 0; j < data.size(); j++) {
                tmap_score.put(data.get(j).value(),data.get(j).name()+" is available in image "+(i+1));
            }

            /*To find top unique label in whole video
             *Treemap data structure eliminate duplicate and its faster to put Key value pair with constant time complexity*/
            for (int j = 0; j < data.size(); j++) {
                tmap_Label.put(data.get(j).name().trim(),data.get(j).value());
                ++labelCount;
            }
            for (int j = 11; j > 0; j--) {
                image.drawText(data.get(j).name(), (int)Math.floor(Math.random()*x), (int)Math.floor(Math.random()*y), HersheyFont.ASTROLOGY, j*5, RGBColour.BLUE);
            }
            DisplayUtilities.displayName(image, "image" + i);
        }


        System.out.println();
        System.out.println("Note: Image is annotated with top 10 Label in decreasing font size");
        System.out.println();

        //Summarising Label
        System.out.println("******** Unique Label summary of Whole video ********************");
        System.out.println();
        Set set2 = tmap_Label.entrySet();

        Iterator itrSet2 = set2.iterator();

        int sNo=0;
        System.out.println("The whole is annotated and summarised all the unique key Label description ");
        while(itrSet2.hasNext()) {
            Map.Entry mEntry2 = (Map.Entry)itrSet2.next();
            System.out.print("("+sNo+") "+mEntry2.getKey() + " ");
            ++sNo;
        }
        System.out.println();
        System.out.println();
        System.out.println("Label Summary :");
        System.out.println("  Out of "+labelCount+" label , there are "+sNo+" unique label");
        System.out.println();

        //Summarising confidence score for whole video inorder of highest to lowest
        System.out.println("******** Consolidated confidence score of all Main frame *******");
        System.out.println();
        Set set = tmap_score.entrySet();

        Iterator itrSet = set.iterator();

        while(itrSet.hasNext()) {
            Map.Entry mEntry = (Map.Entry)itrSet.next();
            System.out.print(mEntry.getKey() + " confidence score of ");
            System.out.println(mEntry.getValue());
        }




    }
}
