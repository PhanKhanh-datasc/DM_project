import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.supervised.instance.Resample; // FIXED: Using supervised Resample filter
import java.io.File;

public class DataPreProcessor {

    // Define constants for the file names
    private static final String CSV_FILE_NAME = "heart_disease.csv";
    private static final String ARFF_FILE_NAME = "heart_disease_processed.arff";
    
    public static void convertCsvToArff(String csvPath, String arffPath) {
        try {
            // Recommended Clean-up step: Delete previous ARFF file
            File arffFile = new File(arffPath);
            if (arffFile.exists()) {
                arffFile.delete();
                System.out.println("Removed existing ARFF file: " + arffPath);
            }

            // 1. Load the CSV file and set class index
            System.out.println("Attempting to load CSV from: " + csvPath);
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvPath));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); 

            // 2. Data Cleaning - ReplaceMissingValues filter (Imputation)
            System.out.println("Applying ReplaceMissingValues filter...");
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(data);
            Instances processedData = Filter.useFilter(data, replaceMissing);
            
            // 3. Class Balancing - Resample filter (FIXED and Working)
            // -Z 100 = Sample size percentage 100
            // -B 1.0 = Bias to uniform class distribution (now correctly recognized)
            System.out.println("Applying Supervised Resample filter (Z=100, Bias=1.0)...");
            Resample resample = new Resample();
            
            // This setOptions now correctly maps to the supervised filter's logic
            resample.setOptions(new String[] {"-Z", "100", "-B", "1.0", "-S", "1"}); 
            
            resample.setInputFormat(processedData);
            Instances finalData = Filter.useFilter(processedData, resample);
            
            System.out.println("Data processed. Original Instances: " + processedData.numInstances() + 
                               ", Resampled Instances: " + finalData.numInstances());

            // 4. Save as ARFF file for later use
            ArffSaver saver = new ArffSaver();
            saver.setInstances(finalData); // Use the resampled data
            saver.setFile(new File(arffPath));
            saver.writeBatch();
            
            System.out.println("----------------------------------------");
            System.out.println("SUCCESS: Processed and Resampled data saved to: " + arffPath);
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