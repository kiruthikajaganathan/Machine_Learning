import moa.core.InstanceExample;
import moa.core.Example;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.streams.ArffFileStream;
import com.yahoo.labs.samoa.instances.Instance;

public class HoeffdingTreeExample {
    public static void main(String[] args) {
        String arffFilePath = "C:\\Program Files\\Weka-3-9-6\\data\\iris.arff";
        ArffFileStream stream = new ArffFileStream(arffFilePath, -1);
        stream.prepareForUse();
        Classifier classifier = new HoeffdingTree();
        classifier.setModelContext(stream.getHeader());
        classifier.prepareForUse();
        BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
        int numberOfInstances = 150;
        for (int i = 0; i < numberOfInstances && stream.hasMoreInstances(); i++) {
            Instance instance = stream.nextInstance().getData();
            Example<Instance> example = new InstanceExample(instance);
            classifier.trainOnInstance(example);
            evaluator.addResult(example, classifier.getVotesForInstance(instance));
            if (i > 0 && i % 50 == 0) {
                System.out.println("Processed " + i + " instances.");
            }
        }
        System.out.println("Accuracy: " + evaluator.getFractionCorrectlyClassified() * 100 + "%");
        System.out.println("Kappa Statistic: " + evaluator.getKappaStatistic());
        System.out.println("Kappa Temporal Statistic: " + evaluator.getKappaTemporalStatistic());
    }
}
