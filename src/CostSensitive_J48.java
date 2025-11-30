import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.CostMatrix; // Explicitly import CostMatrix
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

public class CostSensitive_J48 {

    // Input file: Full processed (and resampled) data for 10-fold CV
    private static final String INPUT_ARFF_FILE = "datasets/heart_disease_processed.arff";
    
    // Output folders
    private static final String MODEL_OUTPUT_FOLDER = "models/";
    private static final String RESULTS_OUTPUT_FOLDER = "results/";
    private static final String RESULTS_FILE = RESULTS_OUTPUT_FOLDER + "evaluation_results.txt"; // Consolidated results file
    
    private static final String CLASSIFIER_NAME = "J48";
    private static final String MODEL_FILE_NAME = "J48_COSTSENSITIVE.model";

    public static void runClassifier() {
        try {
            // Ensure output directories exist
            new File(MODEL_OUTPUT_FOLDER).mkdirs();
            new File(RESULTS_OUTPUT_FOLDER).mkdirs();

            System.out.println("\n--- Starting COST-SENSITIVE " + CLASSIFIER_NAME + " (10-Fold CV) ---");
            DataSource source = new DataSource(INPUT_ARFF_FILE);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            
            // 1. Initialize Base Classifier (J48)
            weka.classifiers.Classifier baseClassifier = new J48();
            
            // 2. Configure and Wrap in CostSensitiveClassifier 
            CostSensitiveClassifier costClassifier = new CostSensitiveClassifier();
            costClassifier.setClassifier(baseClassifier);
            
            // FIX: Correctly instantiate and set the CostMatrix
            // The class index is the last attribute. Since the classes are binary (No, Yes),
            // No is index 0, Yes is index 1.
            CostMatrix costMatrix = new CostMatrix(2); 
            // setCell(actual_class_index, predicted_class_index, cost)
            
            // Cost of False Positive (FP): Actual No (0) -> Predicted Yes (1) = 1.0
            costMatrix.setCell(0, 1, 1.0); 
            
            // Cost of False Negative (FN): Actual Yes (1) -> Predicted No (0) = 3.0 (3x penalty)
            costMatrix.setCell(1, 0, 3.0); 
            
            costClassifier.setCostMatrix(costMatrix);
            
            weka.classifiers.Classifier finalClassifier = costClassifier;
            String modelFilePath = MODEL_OUTPUT_FOLDER + MODEL_FILE_NAME;

            // 3. Model Training and Run-time Measurement
            long startTime = System.currentTimeMillis();
            finalClassifier.buildClassifier(data);
            long endTime = System.currentTimeMillis();
            long buildTime = endTime - startTime;

            // 4. Evaluate using 10-fold cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(finalClassifier, data, 10, new Random(1)); 
            
            // 5. Save Model to the models/ folder
            SerializationHelper.write(modelFilePath, finalClassifier);
            System.out.println("Model saved as: " + modelFilePath);
            
            // 6. Save Results to File
            saveResultsToFile("Cost-Sensitive " + CLASSIFIER_NAME + " (10-Fold CV)", eval, buildTime);

        } catch (Exception e) {
            System.err.println("Error during Cost-Sensitive J48 classification:");
            e.printStackTrace();
        }
    }

    // Helper function to append results to the shared file (unchanged)
    private static void saveResultsToFile(String algorithmName, Evaluation eval, long buildTime) throws Exception {
        StringBuilder results = new StringBuilder();
        results.append("\n============================================\n");
        results.append("RESULTS FOR: ").append(algorithmName).append("\n");
        results.append("Evaluation Method: 10-Fold Cross-Validation\n");
        results.append("Cost Matrix Used: [0, 1.0; 3.0, 0] (3:1 FN:FP)\n");
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
            new File(RESULTS_OUTPUT_FOLDER).mkdirs();
            new File(RESULTS_FILE).delete();
        } catch (Exception e) {
            // Ignore error if file doesn't exist
        }
        
        runClassifier(); 
    }
}