package alaic.alquerier;

import alaic.alquerier.*;
import java.util.*;
import java.io.*;
import alaic.alquerier.Misc;
import weka.core.Instance;
import weka.core.Instances;

//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//


public class CalDensities {

	public static HashMap densityHash = new HashMap();
	

	public static void main (String[] args) throws FileNotFoundException, Exception {
	}
		
	public static HashMap computeDensities(Instances ilist) throws Exception {
		densityHash.clear();
		
		for (int i=0; i<ilist.size(); i++) {
			Instance inst1 = (Instance)ilist.get(i);
			for (int j=0; j<=i; j++) {
				Instance inst2 = (Instance)ilist.get(j);
				
				double sim = 0.0;
				
				sim = Misc.euclideanSimilarity(inst1, inst2, ilist);
				
				updateHash(inst1, sim/ilist.size());
				if (i != j)
					updateHash(inst2, sim/ilist.size());
				if (i % 1000 == 0) System.out.println("Process " + i*j + " of " + (ilist.size()*ilist.size())/2);
			}
			
		}
		
		return densityHash;
	}

	/**
	 * updateHash
	 */
	public static void updateHash(Instance inst, double val) {
		if (densityHash.containsKey(inst)) {
			double oldVal = ((Double)densityHash.get(inst)).doubleValue();
			densityHash.put(inst, new Double(val + oldVal));
		}
		else
			densityHash.put(inst, new Double(val));
	}

}