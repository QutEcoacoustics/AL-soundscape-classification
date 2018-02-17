
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;

import alaic.learner.WekaLearnerOptions;

/**
 * Non-public helper class for Classifier
 * 
 *
 */
class ClassifierUtils {

  public static void updateLearnerOptions(CommandLine commandLine,
      WekaLearnerOptions defaultOptions) throws Exception {

    if(commandLine.hasOption("threads")) {
      defaultOptions.put(defaultOptions.NUM_THREADS, 
          Integer.parseInt(commandLine.getOptionValue("threads")));
    }
    
  }

  public static Options buildOptions() {
    Options options = new Options();
    options.addOption("train", true, "Location of the training file");
    options.addOption("test", true, "Location of the test file");
    options.addOption("evaluate", false, "Used with 'test'. Performs precision recall calculation. Optional.");
    options.addOption("model", true, "Location of the model file. Required.");
    options.addOption("threads", true, "Number of threads to use for parallel training. Default 1");
    options.addOption("help", false, "Display this help");
    
    
    return options;
  }
  
}
