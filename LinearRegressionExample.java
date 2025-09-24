import java.io.File;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;

public class LinearRegressionExample {

    public static void main(String[] args) {
        try {
            // Load the dataset (ARFF file)
            DataSource source = new DataSource("C:\\Users\\kirut\\Downloads\\exam_scores_linearRegression.arff");
            Instances dataset = source.getDataSet();

            // Set the class index to the last attribute (the dependent variable)
            dataset.setClassIndex(dataset.numAttributes() - 1);

            // Create and build the LinearRegression model
            LinearRegression lr = new LinearRegression();
            lr.buildClassifier(dataset);

            // Print the linear regression model
            System.out.println("Linear Regression Model:\n" + lr);

            // Evaluate the model using 10-fold cross-validation
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(lr, dataset, 5, new Random(1));

            // Print evaluation results
            System.out.println("\nEvaluation Results:");
            System.out.println("Mean Absolute Error (MAE): " + eval.meanAbsoluteError());
            System.out.println("Root Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            System.out.println("Root Relative Squared Error: " + eval.rootRelativeSquaredError());
            System.out.println("Relative Absolute Error: " + eval.relativeAbsoluteError());
            System.out.println("Correlation Coefficient: " + eval.correlationCoefficient());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
