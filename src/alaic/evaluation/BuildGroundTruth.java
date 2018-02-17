package alaic.evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.Instances;

public class BuildGroundTruth {

	public static void main(String[] args) throws Exception {
		
		String[] labels= {"insects","birds","fairly quiet","very quiet","light wind","light rain","cicadas","moderate rain","planes","moderate wind","morning chorus","strong wind","loud cicadas"};
		String line = "";
		
		BufferedReader br = new BufferedReader(
						new FileReader("multisource_clusters_true_labels.csv"));
		while ((line = br.readLine()) != null) {
			String[] labs= new String[13];
			String[] lab = line.split(",");
			System.out.println(lab);
		}
	}
}
