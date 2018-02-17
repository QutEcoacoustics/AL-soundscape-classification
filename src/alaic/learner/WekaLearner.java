package alaic.learner;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import alaic.learner.OutputCallback;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;



/**
 * An implementation of the Learner interface using WEKA
 * 
 *
 */
public class WekaLearner implements Learner {

	  boolean learnerInitialized = false;
	  Classifier cls = null;
	  private WekaLearnerOptions _learnerOptions = null;

	  public WekaLearner(WekaLearnerOptions learnerOptions) throws Exception{
	    _learnerOptions = learnerOptions;
	  }

	  public WekaLearnerOptions getLearnerOptions() {
	    return _learnerOptions;
	  }


	  public void saveModel(String modelFileName) throws Exception {
	    if(cls == null) throw new Exception("Model not trained. Nothing to save");
	    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFileName));
	    oos.writeObject(cls);
	    oos.flush();
	    oos.close();
	  }

	 
	  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  1 Here the trainingData is now ready for training
	  public void train(Instances trainingData) throws Exception {
	    train(trainingData, (Double)_learnerOptions.get(_learnerOptions.TRAIN_RATIO));  
	  }

	  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  2
	 
	  public void train(Instances trainingData, double splitRatio) throws Exception {
	    Instances[] split = null;   // create split in type of InstanceList
	    if(splitRatio == 1.0)
	      split = new Instances [] { trainingData, null}; // split[0] is equal to all the trainingData and split[1] is null
	  }
	 


	  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   3
	  
	  public void train(Instances trainingData, Instances evaluationData) throws Exception {
		  
		  int numThreads = (Integer) _learnerOptions.get(_learnerOptions.NUM_THREADS);  // for parallel processing
		  
		  if(numThreads == 1) {
			  
			  Classifier clstrainer = new J48(); 
			  clstrainer.buildClassifier(trainingData);
			  
			  
		  }
	  }
	  
	  public void loadModel(String modelFileName) throws Exception {
	    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFileName));
	    cls= (Classifier)ois.readObject();
	    ois.close();
	    learnerInitialized = true;
	  }

	  private Instances loadTestData(String testFileName) throws Exception {
	    if(cls == null) throw new Exception("Model not trained/loaded");
	    DataSource testFile = new DataSource(testFileName);
		Instances testData = testFile.getDataSet();
		
		
		// setting class attribute if the data format does not provide this information
		// For example, the XRFF format saves the class attribute information as well
		if (testData.classIndex() == -1)
			testData.setClassIndex(testData.numAttributes() - 1);
	    return testData;
	    
	    
	  }

	  @SuppressWarnings("unchecked")
	  public void classify(String testFileName, OutputCallback outputCallback) throws Exception {
	    Instances testData = loadTestData(testFileName);
	    for (int i = 0; i < testData.size(); i++) {
	      Instance input = (Instance)testData.get(i);
	      double output = cls.classifyInstance(input);
	      outputCallback.process(input, output);		  

	      
	    }
	  }

	  
	  public Classifier getModel() {
	    return cls;
	  }
	  

	}

