## Folder Structure

The workspace contains two folders by default, where:
- `bin`: the folder to maintain compiled file
- `datasets`: the folder to maintain data
- `lib`: the folder to maintain dependencies
- `model`: the folder to maintain algorithms code
- `results`: the folder to maintain final result
- `src`: the folder to maintain sources

Meanwhile, the compiled output files will be generated in the `bin` folder by default.

> If you want to customize the folder structure, open `.vscode/settings.json` and update the related settings there.

## Instruction
1. If you wanna run the model, compile all the java first using:
javac -cp ".;lib/weka.jar" -d bin src/*.java

2. How to run the code (from the project directory - root directory)
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

NOTICE: before running SVM_And_KNN.java run CostSensitive_J48.java first, otherwise it will delete the evaluation result text file and write again!!!

