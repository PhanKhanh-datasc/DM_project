import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO; // Support Vector Machine
import weka.classifiers.lazy.IBk; // k-Nearest Neighbors
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
    private static final String MODEL_OUTPUT_FOLDER = "models/"; // Models go here
    private static final String RESULTS_OUTPUT_FOLDER = "results/"; // Text results go here
    private static final String RESULTS_FILE = RESULTS_OUTPUT_FOLDER + "SVM_KNN_evaluation_results.txt"; // Consolidated results file

    public static void runClassifier(String classifierName) {
        try {
            // Ensure both output directories exist
            new File(MODEL_OUTPUT_FOLDER).mkdirs();
            new File(RESULTS_OUTPUT_FOLDER).mkdirs();
            
            // 1. Load Scaled Data Split
            System.out.println("\n--- Starting " + classifierName + " (Test Set Evaluation) ---");
            
            DataSource trainSource = new DataSource(TRAIN_FILE);
            Instances trainData = trainSource.getDataSet();
            trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource testSource = new DataSource(TEST_FILE);
            Instances testData = testSource.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);
            
            weka.classifiers.Classifier classifier;
            String modelFileName;

            // 2. Initialize Classifier
            if ("SMO".equals(classifierName)) {
                classifier = new SMO();
                modelFileName = "SMO_SVM.model";
            } else if ("IBk".equals(classifierName)) {
                IBk knn = new IBk();
                knn.setOptions(new String[] {"-K", "5"}); 
                classifier = knn;
                modelFileName = "IBk_KNN.model";
            } else {
                System.err.println("Unknown classifier: " + classifierName);
                return;
            }
            
            // Generate the full path for the model file
            String modelFilePath = MODEL_OUTPUT_FOLDER + modelFileName; // <--- MODIFIED PATH

            // 3. Model Training and Run-time Measurement (on Training Data)
            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(trainData);
            long endTime = System.currentTimeMillis();
            long buildTime = endTime - startTime;

            // 4. Evaluate on Separate Test Set
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(classifier, testData);
            
            // 5. Save Model to the models/ folder
            SerializationHelper.write(modelFilePath, classifier);
            System.out.println("Model saved as: " + modelFilePath);
            
            // 6. Save Results to File
            saveResultsToFile(classifierName + " (Test Set)", eval, buildTime);

        } catch (Exception e) {
            System.err.println("Error during " + classifierName + " classification:");
            e.printStackTrace();
        }
    }

    // Helper function to append results to the shared file
    private static void saveResultsToFile(String algorithmName, Evaluation eval, long buildTime) throws Exception {
        // ... (results string formatting is unchanged) ...
        StringBuilder results = new StringBuilder();
        results.append("\n============================================\n");
        results.append("RESULTS FOR: ").append(algorithmName).append("\n");
        results.append("Evaluation Method: Test Set Evaluation\n");
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