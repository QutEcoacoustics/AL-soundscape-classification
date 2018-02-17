package alaic.learner;

import java.util.HashMap;

/**
 * A table of options specific to Learner
 * 
 *
 */
public class LearnerOptions extends HashMap<String, Object>{
 
  
  /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
  public static final String MAX_ITERATIONS = "SL_MAX_ITERATIONS";
  public static final String TRAIN_RATIO = "SL_TRAIN_RATIO";
  public static final String NUM_THREADS = "SL_NUM_THREADS";
  public static final String AUX_TRAIN_RATIO = "SL_AUX_TRAIN_RATIO";
  
  public LearnerOptions() {
    initializeOptionDefaults();
  }

  private void initializeOptionDefaults() {
    
    // set the options to their defaults
    put(MAX_ITERATIONS, 400);   
    put(NUM_THREADS, 1);
    put(AUX_TRAIN_RATIO, new Double(1.0));
  }
}
