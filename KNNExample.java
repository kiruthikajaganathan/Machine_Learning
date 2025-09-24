import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KNNExample {

    public static void main(String[] args) {
        try {
            // Load dataset from ARFF file
            DataSource source = new DataSource("C:/Users/kirut/Documents/weka/iris.arff");
            Instances data = source.getDataSet();

            // Set class index (the attribute to predict)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Initialize the k-NN classifier with k=3
            IBk knn = new IBk(3);

            // Build the classifier
            knn.buildClassifier(data);

            // Print the k-NN model summary
            System.out.println("k-NN Model:");
            System.out.println(knn);

            // Evaluate the model using 10-fold cross-validation
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(knn, data, 10, data.getRandomNumberGenerator(1));

            // Print evaluation results
            System.out.println("\nEvaluation Results:");
            System.out.println("Correctly Classified Instances: " + evaluation.pctCorrect() + "%");
            System.out.println("Incorrectly Classified Instances: " + evaluation.pctIncorrect() + "%");
            System.out.println("Kappa Statistic: " + evaluation.kappa());
            System.out.println("Mean Absolute Error: " + evaluation.meanAbsoluteError());
            System.out.println("Root Mean Squared Error: " + evaluation.rootMeanSquaredError());
            System.out.println("Relative Absolute Error: " + evaluation.relativeAbsoluteError() + "%");
            System.out.println("Root Relative Squared Error: " + evaluation.rootRelativeSquaredError() + "%");
            
            // Print confusion matrix
            System.out.println("\nConfusion Matrix:");
            double[][] confusionMatrix = evaluation.confusionMatrix();
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    System.out.print(confusionMatrix[i][j] + " ");
                }
                System.out.println();
            }

            // Print detailed per-class statistics
            System.out.println("\nDetailed per-class statistics:");
            for (int i = 0; i < data.numClasses(); i++) {
                System.out.println("Class " + i + ": " + data.classAttribute().value(i));
                System.out.println("True positives: " + evaluation.numTruePositives(i));
                System.out.println("False negatives: " + evaluation.numFalseNegatives(i));
                System.out.println("False positives: " + evaluation.numFalsePositives(i));
                System.out.println("True negatives: " + evaluation.numTrueNegatives(i));
                System.out.println("Precision: " + evaluation.precision(i));
                System.out.println("Recall: " + evaluation.recall(i));
                System.out.println("F-measure: " + evaluation.fMeasure(i));
                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
