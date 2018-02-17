package alaic.evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import meka.core.MLEvalUtils;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class MultiLabelEval {

	public static void main(String[] args) throws Exception {

		Instances unlabeled = new Instances(
				new BufferedReader(
						new FileReader("multisource_clusters_evaluation_set.arff")));
		
		String line = "";
		String[][] trueLabs = new String [unlabeled.size()][];
		// READ GROUND TRUTH FILE HERE
		int Y[][] = new int[unlabeled.size()][13];
		// read csv file that contains the true labels for each instance in separate lines 
		BufferedReader br = new BufferedReader(
				new FileReader("multisource_clusters_true_labels.csv"));
		//make a list of true labels
		int ind = 0;
		while ((line = br.readLine()) != null) {
			String[] lab = line.split(",");
			trueLabs[ind] =lab;
			ind++;
		}

		// for each instance, make an array of corresponding true labels with length 13, e.g., [0,1,0,...] 
		String[] labels= {"insects","birds","fairly quiet","very quiet","light wind","light rain","cicadas","moderate rain","planes","moderate wind","morning chorus","strong wind","loud cicadas"};

		for (int i = 0; i < unlabeled.numInstances(); i++) {
			int [] indLabels = new int [13] ;
			for (int j = 0; j < labels.length; j++) {
				for (String k : trueLabs[i]) {
					if (k.equals(labels[j])) {
						indLabels[j]= 1;
						break;
					}
				}
				if 	(indLabels[j] != 1) {
					indLabels[j] = 0;
				}
			}
			Y[i] = indLabels;
		}

		double[][] allpreds = new double[unlabeled.size()][Y[0].length];

		// set class index
		if (unlabeled.classIndex() == -1)
			unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
		//load the model
		RandomForest classifier = (RandomForest) weka.core.SerializationHelper.read("ALmodel.model"); //

		//make predictions 
		for (int i = 0; i < unlabeled.numInstances(); i++) {
			Instance inst = unlabeled.instance(i);
			allpreds [i] = classifier.distributionForInstance(inst);
		}

		MLEvalUtils mleval = new MLEvalUtils() {};

		HashMap<String,Object> results = mleval.getMLStats(allpreds, Y, "0.1", "5");

		for (String metric : results.keySet()){
			System.out.println(metric +  "\t" + results.get(metric));
		}

	}

}
