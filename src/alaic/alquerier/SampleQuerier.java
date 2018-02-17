package alaic.alquerier;

import weka.classifiers.Classifier;
import weka.core.Instances;




// returns a list of IDs in poolData to be queries

public abstract class SampleQuerier {
	public abstract int[] select(Classifier model, Instances poolData, Instances trainData, int num);
}
