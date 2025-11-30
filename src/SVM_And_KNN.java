import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO; // Support Vector Machine
import weka.classifiers.lazy.IBk; // k-Nearest Neighbors
import weka.classifiers.meta.CostSensitiveClassifier; 
import weka.classifiers.CostMatrix; // Explicitly import CostMatrix
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class SVM_And_KNN {

    // Input files: Scaled split data (produced by ScaledDataSplitter.java)
    private static final String TRAIN_FILE = "datasets/training_scaled.arff";
    private static final String TEST_FILE = "datasets/testing_scaled.arff";
    
    // Output folders
    private static final String MODEL_OUTPUT_FOLDER = "models/";
    private static final String RESULTS_OUTPUT_FOLDER = "results/";
    private static final String RESULTS_FILE = RESULTS_OUTPUT_FOLDER + "evaluation_results.txt";

    public static void runClassifier(String classifierName) {
        try {
            new File(MODEL_OUTPUT_FOLDER).mkdirs();
            new File(RESULTS_OUTPUT_FOLDER).mkdirs();
            
            // 1. Load Scaled Data Split
            System.out.println("\n--- Starting COST-SENSITIVE " + classifierName + " (Test Set Evaluation) ---");
            
            DataSource trainSource = new DataSource(TRAIN_FILE);
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource testSource = new DataSource(TEST_FILE);
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);
            
            weka.classifiers.Classifier baseClassifier;
            String modelFileName;

            // 2. Initialize Base Classifier
            if ("SMO".equals(classifierName)) {
                baseClassifier = new SMO();
                modelFileName = "SMO_COSTSENSITIVE.model";
            } else if ("IBk".equals(classifierName)) {
                IBk knn = new IBk();
                knn.setOptions(new String[] {"-K", "5"}); 
                baseClassifier = knn;
                modelFileName = "IBk_COSTSENSITIVE.model";
            } else {
                System.err.println("Unknown classifier: " + classifierName);
                return;
            }
            
            // 3. Configure and Wrap in CostSensitiveClassifier 
            CostSensitiveClassifier costClassifier = new CostSensitiveClassifier();
            costClassifier.setClassifier(baseClassifier);
            
            // FIX: Correctly instantiate and set the CostMatrix
            CostMatrix costMatrix = new CostMatrix(2); 
            costMatrix.setCell(0, 1, 1.0); // FP Cost = 1.0
            costMatrix.setCell(1, 0, 3.0); // FN Cost = 3.0
            
            costClassifier.setCostMatrix(costMatrix);
            
            weka.classifiers.Classifier finalClassifier = costClassifier;
            String modelFilePath = MODEL_OUTPUT_FOLDER + modelFileName;

            // 4. Model Training and Run-time Measurement (on Training Data)
            long startTime = System.currentTimeMillis();
            finalClassifier.buildClassifier(trainData);
            long endTime = System.currentTimeMillis();
            long buildTime = endTime - startTime;

            // 5. Evaluate on Separate Test Set
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(finalClassifier, testData);
            
            // 6. Save Model to the models/ folder
            SerializationHelper.write(modelFilePath, finalClassifier);
            System.out.println("Model saved as: " + modelFilePath);
            
            // 7. Save Results to File
            saveResultsToFile("Cost-Sensitive " + classifierName + " (Test Set)", eval, buildTime);

        } catch (Exception e) {
            System.err.println("Error during " + classifierName + " classification:");
            e.printStackTrace();
        }
    }

    // Helper function to append results to the shared file (unchanged)
    private static void saveResultsToFile(String algorithmName, Evaluation eval, long buildTime) throws Exception {
        StringBuilder results = new StringBuilder();
        results.append("\n============================================\n");
        results.append("RESULTS FOR: ").append(algorithmName).append("\n");
        results.append("Evaluation Method: Test Set Evaluation\n");
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
        // Run Algorithms
        runClassifier("SMO"); 
        runClassifier("IBk"); 
    }
}