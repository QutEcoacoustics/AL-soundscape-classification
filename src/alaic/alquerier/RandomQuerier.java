package alaic.alquerier;


import java.util.*;
import java.text.DecimalFormat;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;


//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//


public class RandomQuerier extends SampleQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");

	public int[] select(Classifier model, Instances poolData, Instances trainData, int num) {
		Ranker[] scoring = new Ranker[num]; 
		Random r = new Random();
		int[] ret = new int[num];
		for (int i=0; i<ret.length; i++) {
			ret[i] = r.nextInt(poolData.size());
		    Instance inst = poolData.get(ret[i]); 
			scoring[i] = new Ranker(ret[i], inst);
			System.out.println("\t"+toString()+"\t"+scoring[i]);
		
		}

		Arrays.sort(ret);
		return ret;
	}

	/**
	 * toString
	 */
	public String toString() {
		return "RANDOM";
	}
	
}
