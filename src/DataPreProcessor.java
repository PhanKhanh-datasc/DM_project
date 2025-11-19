import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import java.io.File;

public class DataPreProcessor {

    // Define constants for the file names
    private static final String CSV_FILE_NAME = "heart_disease.csv";
    private static final String ARFF_FILE_NAME = "heart_disease_processed.arff";
    
    // The core logic for loading, cleaning, and saving
    public static void convertCsvToArff(String csvPath, String arffPath) {
        try {
            // 1. Load the CSV file using the provided absolute or relative path
            System.out.println("Attempting to load CSV from: " + csvPath);
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvPath));
            Instances data = loader.getDataSet();
            System.out.println("Data loaded successfully.");

            // Set the class attribute (target label)
            // Heart Disease Status is the last attribute (index = total attributes - 1)
            data.setClassIndex(data.numAttributes() - 1); 

            // 2. Data Cleaning - ReplaceMissingValues filter
            // Essential for handling the missing entries found in the dataset.
            System.out.println("Applying ReplaceMissingValues filter...");
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(data);
            
            // Apply the filter
            Instances processedData = Filter.useFilter(data, replaceMissing);
            
            System.out.println("Data processed. Saving ARFF to: " + arffPath);

            // 3. Save as ARFF file for later use
            ArffSaver saver = new ArffSaver();
            saver.setInstances(processedData);
            saver.setFile(new File(arffPath)); // Saves to the specified path
            saver.writeBatch();
            
            System.out.println("----------------------------------------");
            System.out.println("SUCCESS: Processed data saved to: " + arffPath);
            System.out.println("Total Instances: " + processedData.numInstances());
            System.out.println("----------------------------------------");
            
        } catch (Exception e) {
            System.err.println("Error during data processing:");
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        String csvPath = "datasets/" + CSV_FILE_NAME;
        String arffPath = "datasets/" + ARFF_FILE_NAME;
        convertCsvToArff(csvPath, arffPath);
    }
}