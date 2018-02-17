package alaic.learner;

import weka.core.Instances;
import weka.classifiers.Classifier;




/**
 * Interface for learning
 *
 */
public interface Learner {
  /**
   * Train a model from the training data
   * @param trainingData
   * @throws Exception
   */
  public void train(Instances trainingData) throws Exception;
  
  
  /**
   * 
   * @param trainingData
   * @param evaluationData
   * @throws Exception
   */
  public void train(Instances trainingData, Instances evaluationData) throws Exception;
  
  /**
   * Save a trained model to a file specified by modelFileName 
   * @param modelFileName
   * @throws Exception
   */
  public void saveModel(String modelFileName) throws Exception;
  
  /**
   * Load a trained model from the file specified by modelFileName
   * @param modelFileName
   * @throws Exception
   */
  public void loadModel(String modelFileName) throws Exception;
  
  /**
   * Classifies instances in the testFileName. For each classification result:
   * outputCallback.process() is called
   * @param testFileName
   * @param outputCallback
   * @throws Exception
   */
  public void classify(String testFileName, OutputCallback outputCallback) throws Exception;
  
  
  /**
   * 
   * @return learnerOptions
   */
  public LearnerOptions getLearnerOptions();

  /**
   * If the model is trained, return the pip from the model
   * @return
   * @throws Exception
   */


  /**
   * Return transducer model
   * @return
   */
  public Classifier getModel();

}
