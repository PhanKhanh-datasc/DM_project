import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class ClassificationEngine {
    private static final String ARFF_FILE = "datasets/heart_disease_processed.arff";

    public static void runClassifier(String classifierName) {
        try {
            // 1. Load the processed ARFF data
            System.out.println("Loading data from: " + ARFF_FILE);
            DataSource source = new DataSource(ARFF_FILE);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            
            weka.classifiers.Classifier classifier;
            String modelFileName;

            // 2. Initialize the correct classifier
            if ("J48".equals(classifierName)) {
                // Chapter 3: Implementation of Algorithm 1
                classifier = new J48();
                modelFileName = "J48_DECISIONTREE.model";
                System.out.println("\n--- Starting J48 Decision Tree (Baseline) ---");
            } else if ("RandomForest".equals(classifierName)) {
                // Chapter 4: Implementation of Algorithm 2 (Improvement)
                classifier = new weka.classifiers.trees.RandomForest();
                modelFileName = "RANDOMFOREST.model";
                System.out.println("\n--- Starting Random Forest (Improvement) ---");
            } else {
                System.err.println("Unknown classifier: " + classifierName);
                return;
            }

            // 3. Model Training and Run-time Measurement
            // This is required by the assignment hint.
            long startTime = System.currentTimeMillis();
            classifier.buildClassifier(data);
            long endTime = System.currentTimeMillis();
            System.out.println("Model Building Run-time: " + (endTime - startTime) + " ms");
            
            // 4. Evaluate using 10-fold cross-validation (required)
            System.out.println("Evaluating model using 10-fold cross-validation...");
            Evaluation eval = new Evaluation(data);
            // Use a fixed seed (1) for reproducible cross-validation results
            eval.crossValidateModel(classifier, data, 10, new Random(1)); 
            
            // 5. Output Results (for Chapter 5)
            System.out.println(eval.toSummaryString("\nResults Summary\n===============\n", false));
            System.out.println(eval.toClassDetailsString("\nDetailed Class Performance:\n"));
            System.out.println(eval.toMatrixString("\nConfusion Matrix:\n"));
            System.out.printf("Overall Accuracy: %.2f%%\n", eval.pctCorrect());

            // 6. Save the trained model to a binary file (as required)
            SerializationHelper.write(modelFileName, classifier);
            System.out.println("\nSuccessfully saved model to: " + modelFileName);

        } catch (Exception e) {
            System.err.println("Error during classification:");
            e.printStackTrace();
        }
    }
    public static void main(String[] args) {
        // Run Algorithm 1
        runClassifier("J48"); 
        
        // Run Algorithm 2
        runClassifier("RandomForest"); 
    }
    
}
