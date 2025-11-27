import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

public class J48_And_RandomForest {

    // Input file: Full processed data for 10-fold CV
    private static final String INPUT_ARFF_FILE = "datasets/heart_disease_processed.arff";
    
    // Output folders
    private static final String MODEL_OUTPUT_FOLDER = "models/"; // Models go here
    private static final String RESULTS_OUTPUT_FOLDER = "results/"; // Text results go here
    private static final String RESULTS_FILE = RESULTS_OUTPUT_FOLDER + "J48_RF_evaluation_results.txt"; // Consolidated results file

    public static void runClassifier(String classifierName) {
        try {
            // Ensure both output directories exist
            new File(MODEL_OUTPUT_FOLDER).mkdirs();
            new File(RESULTS_OUTPUT_FOLDER).mkdirs();

            // 1. Load Data
            System.out.println("\n--- Starting " + classifierName + " (10-Fold CV) ---");
            DataSource source = new DataSource(INPUT_ARFF_FILE);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            
            weka.classifiers.Classifier classifier;
            String modelFileName;

            // 2. Initialize Classifier
            if ("J48".equals(classifierName)) {
                classifier = new J48();
                modelFileName = "J48_DECISIONTREE.model";
            } else if ("RandomForest".equals(classifierName)) {
                classifier = new RandomForest();
                modelFileName = "RANDOMFOREST.model";
            } else {
                System.err.println("Unknown classifier: " + classifierName);
                return;
            }
            
            // Generate the full path for the model file
            String modelFilePath = MODEL_OUTPUT_FOLDER + modelFileName; // <--- MODIFIED PATH

            // 3. Model Training and Run-time Measurement
            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(data);
            long endTime = System.currentTimeMillis();
            long buildTime = endTime - startTime;

            // 4. Evaluate using 10-fold cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(1)); 
            
            // 5. Save Model to the models/ folder
            SerializationHelper.write(modelFilePath, classifier);
            System.out.println("Model saved as: " + modelFilePath);
            
            // 6. Save Results to File
            saveResultsToFile(classifierName + " (10-Fold CV)", eval, buildTime);

        } catch (Exception e) {
            System.err.println("Error during J48/RF classification:");
            e.printStackTrace();
        }
    }

    // Helper function to append results to the shared file
    private static void saveResultsToFile(String algorithmName, Evaluation eval, long buildTime) throws Exception {
        // ... (results string formatting is unchanged) ...
        StringBuilder results = new StringBuilder();
        results.append("\n============================================\n");
        results.append("RESULTS FOR: ").append(algorithmName).append("\n");
        results.append("Evaluation Method: 10-Fold Cross-Validation\n");
        results.append("============================================\n");
        results.append("Model Building Run-time: ").append(buildTime).append(" ms\n");
        results.append(eval.toSummaryString("\nResults Summary\n", false));
        results.append(eval.toClassDetailsString("\nDetailed Class Performance:\n"));
        results.append(eval.toMatrixString("\nConfusion Matrix:\n"));
        results.append("Overall Accuracy: ").append(String.format("%.2f%%", eval.pctCorrect())).append("\n");

        try (FileWriter fw = new FileWriter(RESULTS_FILE, true);
             PrintWriter pw = new PrintWriter(fw)) {
            pw.print(results.toString());
            System.out.println("Metrics appended to: " + RESULTS_FILE);
        }
    }

    public static void main(String[] args) {
        // Clear previous results file for a clean start
        try {
            // Ensure results folder exists before attempting to delete the file
            new File(RESULTS_OUTPUT_FOLDER).mkdirs();
            new File(RESULTS_FILE).delete();
        } catch (Exception e) {
            // Ignore error if file doesn't exist
        }
        
        // Run Algorithms
        runClassifier("J48"); 
        runClassifier("RandomForest"); 
    }
}