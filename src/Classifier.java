
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;

import alaic.learner.WekaLearner;
import alaic.learner.WekaLearnerOptions;
import alaic.learner.LearnerOptions;
import alaic.learner.SupALTrainer;



/**
 * Classifier
 * 
 *
 */
@SuppressWarnings("unused")
public class Classifier {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		CommandLineParser parser = new GnuParser();
		Options options = ClassifierUtils.buildOptions();
		CommandLine commandLine = parser.parse(options, args);
		execute(options, commandLine);
	}

	private static void execute(Options options, CommandLine commandLine) throws Exception {

		if(commandLine.hasOption("help") || commandLine.getOptions().length == 0) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "Classifier", options);
			System.exit(-1);
		}
		if(commandLine.hasOption("model") == false) 
			throw new Exception("Model file should be specified.");

		WekaLearnerOptions defaultOptions = new WekaLearnerOptions();
		WekaLearner learner = new WekaLearner(defaultOptions);

		if(commandLine.hasOption("train")) {
			
			SupALTrainer trainer = new SupALTrainer(learner);
			trainer.ALandSupWholeTrain(commandLine.getOptionValue("train"));
			
		}
	}
}
