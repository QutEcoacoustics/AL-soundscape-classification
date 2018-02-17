package alaic.alquerier;

import alaic.*;
import java.util.*;
import java.text.DecimalFormat;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//


public abstract class InformationDensityQuerier extends SampleQuerier { 
    private static DecimalFormat DF = new DecimalFormat("####.######");
    private static HashMap HASH = null;
    private static double BETA = 1.0;

    public InformationDensityQuerier(HashMap densityHash) {
        HASH = densityHash;
    }


    public int[] select(Classifier model, Instances resampledData, Instances trainData, int num) { 
        int[] ret = new int[num];
        double[] scores = new double[resampledData.size()];
        Ranker[] scoring = new Ranker[resampledData.size()];
        Arrays.fill(scores, 0.0);
        
        for (int i=0; i<resampledData.size(); i++) {
            Instance inst = (Instance)resampledData.get(i);
            double prob = Misc.getprobability(model, inst);
            double dens = ((Double)HASH.get(inst)).doubleValue();
            scores[i] = (1-prob) * Math.pow(dens, BETA); 
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
        return "IDen";
    }


}
    

