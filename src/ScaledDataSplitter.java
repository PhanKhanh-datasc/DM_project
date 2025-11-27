import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import java.io.File;
import java.util.Random;

public class ScaledDataSplitter {

    // Input file from DataPreProcessor
    private static final String INPUT_ARFF_FILE = "datasets/heart_disease_processed.arff";
    
    // Output files for scaled classification (SVM/k-NN)
    private static final String TRAIN_SCALED_FILE = "datasets/training_scaled.arff";
    private static final String TEST_SCALED_FILE = "datasets/testing_scaled.arff";
    
    // Split parameters
    private static final double TRAIN_RATIO = 0.7;
    private static final long RANDOM_SEED = 42; 

    public static void processAndSplitData() {
        try {
            // 1. Load the fully processed ARFF data
            System.out.println("Loading cleaned data from: " + INPUT_ARFF_FILE);
            DataSource source = new DataSource(INPUT_ARFF_FILE);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); 
            
            // 2. Shuffle the data
            System.out.println("Shuffling data...");
            Instances transformedData = new Instances(data); // Create a mutable copy
            transformedData.randomize(new Random(RANDOM_SEED)); 
            
            // 3. Apply Filters (Transformation)
            System.out.println("Applying NominalToBinary and Normalize filters...");
            
            // NominalToBinary (converts nominal features for linear models)
            NominalToBinary nomToBinFilter = new NominalToBinary();
            nomToBinFilter.setInputFormat(transformedData);
            transformedData = Filter.useFilter(transformedData, nomToBinFilter);
            
            // Normalize (scales numeric data to [0, 1])
            Normalize normalizeFilter = new Normalize();
            normalizeFilter.setInputFormat(transformedData);
            transformedData = Filter.useFilter(transformedData, normalizeFilter); 
            
            // 4. Perform the 70/30 split on the transformed data
            int numInstances = transformedData.numInstances();
            int trainSize = (int) Math.round(numInstances * TRAIN_RATIO);
            int testSize = numInstances - trainSize;

            Instances train = new Instances(transformedData, 0, trainSize);
            Instances test = new Instances(transformedData, trainSize, testSize);
            
            // 5. Save Scaled Training Data
            ArffSaver trainSaver = new ArffSaver();
            trainSaver.setInstances(train);
            trainSaver.setFile(new File(TRAIN_SCALED_FILE));
            trainSaver.writeBatch();
            
            // 6. Save Scaled Testing Data
            ArffSaver testSaver = new ArffSaver();
            testSaver.setInstances(test);
            testSaver.setFile(new File(TEST_SCALED_FILE));
            testSaver.writeBatch();
            
            System.out.println("----------------------------------------");
            System.out.println("SUCCESS: Scaled data split (70/30) and saved.");
            System.out.println("Scaled Training Instances: " + TRAIN_SCALED_FILE);
            System.out.println("Scaled Testing Instances: " + TEST_SCALED_FILE);
            System.out.println("----------------------------------------");
            
        } catch (Exception e) {
            System.err.println("Error during data processing, scaling, or splitting:");
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        processAndSplitData();
    }
}