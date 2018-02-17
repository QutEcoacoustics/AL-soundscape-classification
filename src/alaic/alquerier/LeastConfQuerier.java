package alaic.alquerier;


import alaic.alquerier.Ranker;
import alaic.alquerier.SampleQuerier;

import java.util.*;
import java.util.Map.Entry;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//


public abstract class LeastConfQuerier extends SampleQuerier {	
	
	public int[] select(Classifier model, Instances poolData, Instances trainData, int num) {
		int[] ret = new int[num];
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = poolData.get(i);
			double[] prediction;
			
			try {
				double predictedClass = model.classifyInstance(inst);
				int predClass = (int) (predictedClass);
				prediction = model.distributionForInstance(inst);
				double score = prediction [predClass];
				scoring[i] = new Ranker(i, score, inst);

			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("Exception happened....");
			}
		
		}
			
		Arrays.sort(scoring); 
		for (int i=0; i<ret.length; i++) {
			System.out.println("\t"+toString()+"\t"+scoring[scoring.length-(1+i)]);
			ret[i] = scoring[scoring.length-(1+i)].getIndex();
		}
		Arrays.sort(ret);
		return ret;
	}
	
	

	/**
	 * toString
	 */
	public String toString() {
		return "LEASTCONF";
	}
	
}
