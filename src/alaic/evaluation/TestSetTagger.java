package alaic.evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import alaic.alquerier.Ranker;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;


/**
 * To test a trained model over the an unlabeled set
 *
 */
public class TestSetTagger {

	/**
	 * @param args
	 * @throws Exception 
	 */
	@SuppressWarnings("rawtypes")
	
	public static void main(String[] args) throws Exception {
		
		
		BufferedWriter resultWriter = new BufferedWriter(new FileWriter("ModelPrediction.csv")); 
		
		//load the unlabeled data
		 Instances unlabeled = new Instances(
                 new BufferedReader(
                   new FileReader("multi_label_evaluation_set.arff")));
		 
		// set class index
		if (unlabeled.classIndex() == -1)
			unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
		//String[] labels =; 

		//load the saved model
		RandomForest classifier = (RandomForest) weka.core.SerializationHelper.read("ALmodel-14740.model"); 

		//make predictions and sort them based on the probabilities
		for (int i = 0; i < unlabeled.numInstances(); i++) {
			Map <Integer, Double> pred = new HashMap<Integer, Double>();
			Instance inst = unlabeled.instance(i);
			double[] prediction = classifier.distributionForInstance(inst);
			for (int j=0; j<prediction.length; j++) {
				pred.put(j, prediction[j]);
			}
			List<Entry<Integer, Double>> list =
		            new LinkedList<Map.Entry<Integer,Double>>(pred.entrySet());
			
			Collections.sort(list, new Comparator<Map.Entry<Integer,Double>>() {
		        public int compare(
		                Entry<Integer, Double> e1,
		                Entry<Integer, Double> e2) {
		            return e2.getValue().compareTo(e1.getValue());
		        }
		    });
			
		   // retrieve the top 3 predictions, including their probabilities and class values and writing them into the file
		   double[] predProb= new double[3];
		   String[] predClass = new String[3];
		   
		   for (int k=0; k<3; k++) {
			   predProb[k] = list.get(k).getValue(); //retrieve the probability of the model for the predicted class
			   predClass[k] = inst.classAttribute().value(list.get(k).getKey());  // retrieve the value of class attribute by index 
		   }
		   
		   resultWriter.write(predClass[0] + "," + predClass[1] + "," +  predClass[2] + ",");
		   resultWriter.write(predProb[0] + "," + predProb[1] + "," +  predProb[2]);
		   resultWriter.write("\n");

		}
		resultWriter.flush();
		resultWriter.close();

	}

}
