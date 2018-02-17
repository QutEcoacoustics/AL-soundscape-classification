package alaic.alquerier;


import java.util.*;
import java.text.*;
import weka.core.Instance;

//
// This class is based on the EMNLP'08 source code developed by Settles & Craven and released under the GPL v3.0.
//

public class Ranker implements Comparable {
	private static DecimalFormat DF = new DecimalFormat("####.####");
	private double score;
	private int index;
	private Instance inst;
	private int id;
	private int seqLen;
	
	public Ranker(int index, Instance inst) {   
		this.index = index;
		this.inst =  inst;
		
    }
	
	
	public Ranker(int index, double score, Instance inst) {
		this.index = index;
        this.score = score;
		this.inst = inst;
    }
    
    public Ranker(int index, int id, Instance inst) {
		this.index = index;
        this.id = id;
		this.inst = inst;
    }
    
    public Ranker(int index, double score, Instance inst, int id) {   
		this.index = index;
        this.score = score;
		this.inst = inst;
		this.id = id;
    }

	/**
	* Comparison between two objects (for the comparable interface).
	*/
	public int compareTo(Object o) {
		Ranker r = (Ranker)o;
		if (score < r.score)
			return 1;
		else if (score > r.score)
			return -1;
		else
			return 0;
	}


	
	public String toString() {   
		return index+"\t "+DF.format(score);
	}
	
	public double getScore() {
		return score;
	}
	
	public int getIndex() {
		return index;
	}
	public Instance getInstance() {
		return inst;
	}
	
	public int getseqID() {
		return id;
	}
	
}
