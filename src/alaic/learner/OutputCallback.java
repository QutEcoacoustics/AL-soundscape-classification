package alaic.learner;

import weka.core.Instance;

/**
 * 
 * 
 *
 */
public interface OutputCallback {
  @SuppressWarnings("unchecked")
  public void process(Instance input, double output);
}
