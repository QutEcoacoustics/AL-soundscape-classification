package alaic.alquerier;

import alaic.alquerier.*;
import java.util.*;
import java.io.*;
import weka.core.Instance;
import weka.core.Instances;


public class CalDiversities {

	public static HashMap diversityHash = new HashMap();

	public static void main (String[] args) throws FileNotFoundException, Exception {
	}

	public static HashMap computeDiversities(Instances lnlist,Instances ilist) {
		diversityHash.clear();
		
		for (int i=0; i<ilist.size(); i++) {
			Instance inst1 = (Instance)ilist.get(i);
			for (int j=0; j<lnlist.size(); j++) {
				Instance inst2 = (Instance)lnlist.get(j);
						double sim = 0.0;
						sim = Misc.euclideanSimilarity(inst1, inst2, lnlist); 
						
						updateHash(inst1, sim/lnlist.size());
			}
		}
		return diversityHash;
	}

	/**
	 * updateHash
	 */
	public static void updateHash(Instance inst, double val) {
		if (diversityHash.containsKey(inst)) {
			double oldVal = ((Double)diversityHash.get(inst)).doubleValue();
			diversityHash.put(inst, new Double(val + oldVal));
		}
		else
			diversityHash.put(inst, new Double(val));
	}

}