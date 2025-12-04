1. If you wanna run the model, compile all the java first using:
javac -cp ".;lib/weka.jar" -d bin src/*.java

2. How to run the code (inside the project directory -  root directory)
Run DataPreProcessor
java -cp "bin;lib/weka.jar" DataPreProcessor

Run CostSensitive_J48:
java -cp "bin;lib/weka.jar" CostSensitive_J48

Run SVM_And_KNN:
java -cp "bin;lib/weka.jar" SVM_And_KNN

Run DataPreProcessor:
java -cp "bin;lib/weka.jar" DataPreProcessor

Run ScaledDataSplitter:
java -cp "bin;lib/weka.jar" ScaledDataSplitter
