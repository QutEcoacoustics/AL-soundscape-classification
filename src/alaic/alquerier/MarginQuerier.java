package alaic.alquerier;


import java.util.*;
import java.text.DecimalFormat;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//


public class MarginQuerier extends SampleQuerier {	
	private static DecimalFormat DF = new DecimalFormat("####.######");
	
	public MarginQuerier(){}
	
	public int[] select(Classifier model, Instances poolData, Instances trainData, int num) {
		int[] ret = new int[num];
		Ranker[] scoring = new Ranker[poolData.size()];
		for (int i=0; i<poolData.size(); i++) {
			Instance inst = (Instance)poolData.get(i);
			double score = Misc.margin(model, inst);
			scoring[i] = new Ranker(i, score, inst);
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
        return "Margin";
    }

}
