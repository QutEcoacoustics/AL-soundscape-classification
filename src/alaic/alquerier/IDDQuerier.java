package alaic.alquerier;

import alaic.*;
import java.util.*;
import java.text.DecimalFormat;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;



public abstract class IDDQuerier extends SampleQuerier { 
    private static DecimalFormat DF = new DecimalFormat("####.######");
    private static HashMap DENHASH = null;
    private static HashMap DIVHASH = null;
    private static double BETA = 1.0;

    public IDDQuerier(HashMap densityHash, HashMap diversityHash) {
        DENHASH = densityHash;
        DIVHASH = diversityHash;
 
    }
    


    public int[] select(Classifier model, Instances poolData, Instances trainData, int num) {
        int[] ret = new int[num];
        double[] scores = new double[poolData.size()];
        Ranker[] scoring = new Ranker[poolData.size()];
        Arrays.fill(scores, 0.0);
        for (int i=0; i<poolData.size(); i++) {
            Instance inst = (Instance)poolData.get(i);
            double prob = Misc.getprobability(model, inst);
            double dens = ((Double)DENHASH.get(inst)).doubleValue();
            double divs = ((Double)DIVHASH.get(inst)).doubleValue(); 
            scores[i] = (1-prob) * Math.pow(dens, BETA) * (1-divs); 
            scoring[i] = new Ranker(i, scores[i], inst);
            
        }
        
        Arrays.sort(scoring); 
        for (int i=0; i<ret.length; i++) {
            System.out.println("\t"+toString()+"\t"+scoring[i]);
            ret[i] = scoring[i].getIndex();
        }
        Arrays.sort(ret);
        return ret;
    }
    
    

    /**
     * toString
     */
    public String toString() {
        return "IDD";
    }


}
    

