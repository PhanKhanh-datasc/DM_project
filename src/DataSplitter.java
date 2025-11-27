import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;

public class DataSplitter {

    // Input file from DataPreProcessor
    private static final String INPUT_ARFF_FILE = "datasets/heart_disease_processed.arff";
    
    // Output files for classification
    private static final String TRAIN_FILE = "datasets/training.arff";
    private static final String TEST_FILE = "datasets/testing.arff";
    
    // Split ratio (70% for training, 30% for testing)
    private static final double TRAIN_RATIO = 0.7;
    private static final long RANDOM_SEED = 42; // Fixed seed for reproducible split

    public static void splitData() {
        try {
            // 1. Load the fully processed ARFF data
            System.out.println("Loading cleaned data from: " + INPUT_ARFF_FILE);
            DataSource source = new DataSource(INPUT_ARFF_FILE);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); 
            
            // 2. Shuffle the data
            // Crucial step to ensure both sets are representative and unbiased.
            System.out.println("Shuffling data for stratified split...");
            data.randomize(new Random(RANDOM_SEED)); 
            
            // 3. Determine split sizes
            int numInstances = data.numInstances();
            int trainSize = (int) Math.round(numInstances * TRAIN_RATIO);
            int testSize = numInstances - trainSize;

            // 4. Create Training and Testing Instances
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            
            // 5. Save Training Data
            ArffSaver trainSaver = new ArffSaver();
            trainSaver.setInstances(train);
            trainSaver.setFile(new File(TRAIN_FILE));
            trainSaver.writeBatch();
            
            // 6. Save Testing Data
            ArffSaver testSaver = new ArffSaver();
            testSaver.setInstances(test);
            testSaver.setFile(new File(TEST_FILE));
            testSaver.writeBatch();
            
            System.out.println("----------------------------------------");
            System.out.println("SUCCESS: Data split (70/30) and saved.");
            System.out.println("Training Instances saved to: " + TRAIN_FILE);
            System.out.println("Testing Instances saved to: " + TEST_FILE);
            System.out.println("Total Instances: " + numInstances);
            System.out.println("----------------------------------------");
            
        } catch (Exception e) {
            System.err.println("Error during data splitting:");
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        splitData();
    }
}