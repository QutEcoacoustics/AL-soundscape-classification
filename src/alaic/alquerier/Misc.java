package alaic.alquerier;

import alaic.*;
import java.text.DecimalFormat;
import weka.classifiers.Classifier;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//


public class Misc {
    public static DecimalFormat DF = new DecimalFormat("####.######");
   


    static public double euclideanSimilarity(Instance sv1, Instance sv2, Instances instances) {
    	EuclideanDistance metric= new EuclideanDistance(instances);
    	metric.setDontNormalize(true);
        double sim = 1/(1+metric.distance(sv1, sv2));
        if (Double.isNaN(sim))
            return 0.0;

        return sim;
    }
    
    static public double getprobability(Classifier model, Instance inst){
    	
    	double[] prediction;
    	double score = 0.0;
		try {
			double predictedClass = model.classifyInstance(inst);
			int predClass = (int) (predictedClass);
			prediction = model.distributionForInstance(inst);
			score = prediction [predClass];

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Exception happened....");
		}
		return score;

    }

    /**
     * entropy
     */
    static public double entropy(Classifier model, Instance inst) {
    	
    	double[] prediction;
    	double score = 0.0;
		try {
			prediction = model.distributionForInstance(inst);
			for (int i=0; i<prediction.length; i++)
				score -= (prediction[i] > 0)
	                ? prediction[i] * Math.log(prediction[i]) 
	                : 0;

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Exception happened....");
		}
		return score;
		
    }

    /**
     * Margin
     */
    public static double margin(Classifier model, Instance inst) {
    	
    	double[] prediction;
    	double highest1 = 0.0;
    	double highest2 = 0.0;
		try {
			double predictedClass = model.classifyInstance(inst);
			prediction = model.distributionForInstance(inst);
			for(double value: prediction){
				if(value>=highest1){
					highest2 = highest1;
					highest1=value;
				
				}else if(value>highest2){
					highest2=value;
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Exception happened....");
		}
		double score = highest1-highest2;
		return score;
        
    }

}
