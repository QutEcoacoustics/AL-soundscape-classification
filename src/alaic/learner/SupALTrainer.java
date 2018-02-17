package alaic.learner;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;


import java.util.Random;
import alaic.learner.Learner;
import alaic.alquerier.LeastConfQuerier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import alaic.alquerier.CalDensities;
import alaic.alquerier.CalDiversities;
import alaic.alquerier.EntropyQuerier;
import alaic.alquerier.IDDQuerier;
import alaic.alquerier.InformationDensityQuerier;
import alaic.alquerier.InformationDiversityQuerier;
import alaic.alquerier.MarginQuerier;
import alaic.alquerier.RandomQuerier;
import alaic.alquerier.Ranker;



/**
 * Trainer class
 *
 *
 */
public class SupALTrainer {

	private Learner _suplearner = null;
	private Learner _allearner = null;
	public Instances Ln = null;
	public SupALTrainer(Learner learner) {
		_suplearner = learner;
		_allearner = learner;
	}
	
	public void ALandSupWholeTrain(String trainingFileName) throws Exception {
	
		
		DataSource trainFile = new DataSource(trainingFileName);
		Instances allData = trainFile.getDataSet();
		
		// setting class attribute if the data format does not provide this information
		// For example, the XRFF format saves the class attribute information as well
		if (allData.classIndex() == -1)
			allData.setClassIndex(allData.numAttributes() - 1);


		//+++++++++++++++ Evaluation Setup +++++++++++++++++++++++//
		String resultsDirectory = "IDen/";
		File fileModel = new File(resultsDirectory);
		fileModel.mkdirs();
		BufferedWriter bw = new BufferedWriter(new FileWriter(resultsDirectory + "/" + "Result.txt"));
		String testFileName = "test_data.csv";
		
		DataSource testFile = new DataSource(testFileName);
		Instances testData = testFile.getDataSet();
		
		// setting class attribute if the data format does not provide this information
		// For example, the XRFF format saves the class attribute information as well
		if (testData.classIndex() == -1)
			testData.setClassIndex(testData.numAttributes() - 1);
		//+++++++++++++++ Evaluation Setup +++++++++++++++++++++++//
		
		// +++++++++++++++++++++++++++++++++++++ train over the whole data (SUPERVISED LEARNER)
		Classifier clstrainer = new RandomForest(); //SMO(); //J48(); //
		clstrainer.buildClassifier(allData);//clstrainer.buildClassifier(resampledI);

		ObjectOutputStream s =
				new ObjectOutputStream(new FileOutputStream(resultsDirectory +"/"+ "SupervisedModel.model"));//(resultsDirectory + "SupervisedModel-rao.model"));
		s.writeObject(clstrainer);
		s.close();
		
		//+++++++++++++++ Evaluation+++++++++++++++++++++++//
	
		Evaluation evaluator = new Evaluation(testData);
		Classifier modelSup = clstrainer;//      
		bw.write("\n\n*******************************Evaluation******************************\n");
		bw.write("SUPERVISED (Trainset size: " + (allData.size()) + " Testset size: " + testData.size() + ")\n");
		evaluator.evaluateModel(modelSup, testData);		
		bw.write(evaluator.toSummaryString("\nResults\n======\n", true));
		bw.write(evaluator.toClassDetailsString("\nResults Per Class\n======\n"));
		bw.write(evaluator.toMatrixString("\nConfusion Matrix\n======\n"));  //output confusion matrix
		bw.write("-------------------------------------------------------\n\n");
		bw.flush();
		//+++++++++++++++ Evaluation+++++++++++++++++++++++//
		
		// +++++++++++++++++++++++++++++++++++++ train over the whole data (SUPERVISED LEARNER)


		//----------------------- Active Learning Part 
		// Initialising the seed set (Ln)
		int initialLnSize =  20; // 20 samples (20 minutes recording) as the initialed labeled set and the AL batch size to be annotated by a human annotator in a reasonable time.
		// number of informative instanced being selected in each iteration of AL
		int bunchSize = 20; 

		Instances Ln = new Instances(allData, 0);
		
		//**************************************** Seed Set ****************************************************//
				
		//+++++++++++++++Randomly selecting the seed set+++++++++++++++//
		Random rd1 = new Random();
		int no;
		Ranker[] scoring = new Ranker[initialLnSize];
		for (int i=0; i < initialLnSize; i++){
			no = (rd1.nextInt(allData.size()));
			Instance inst = allData.get(no);
			scoring[i] = new Ranker(no, inst); 
			Ln.add(allData.remove(no));  
		}
		
		for (int i=0; i<initialLnSize; i++) {
			System.out.println("\t"+toString()+"\t"+scoring[i]);
		}
		
		//+++++++++++++++Randomly selecting the seed set+++++++++++++++//
		
		//------------Analysing the selected sequences & their classes-----------------

		//AL
		int numIns = 0;
		int numBird = 0;  
		int numFquiet = 0;  
		int numVquiet = 0;  
		int numLwind = 0;  
		int numLrain = 0;
		int numCica = 0;
		int numMrain = 0;
		int numPlan = 0;
		int numMwind = 0;
		int numChorus = 0;
		int numSwind = 0;
		int numLcica = 0;
		
		for (int i=0; i<Ln.size(); i++) {
			Instance inst = Ln.get(i);
			//System.out.println(inst.classAttribute());
			//@attribute Class {insects,birds,'fairly quiet','very quiet','light wind','light rain',cicadas,'moderate rain',planes,'moderate wind','morning chorus','strong wind','loud cicadas'}
			String classVal = inst.stringValue(inst.classAttribute());
			if(classVal.equals("insects")){
				numIns = numIns+1;
			}
			if(classVal.equals("birds")){
				numBird = numBird+1;
			}
			if(classVal.equals("fairly quiet")){
				numFquiet = numFquiet+1;
			}
			if(classVal.equals("very quiet")){
				numVquiet = numVquiet+1;
			}
			if(classVal.equals("light wind")){
				numLwind = numLwind+1;
			}
			if(classVal.equals("light rain")){
				numLrain = numLrain+1;
			}
			if(classVal.equals("cicadas")){
				numCica = numCica+1;
			}
			if(classVal.equals("moderate rain")){
				numMrain = numMrain+1;
			}
			if(classVal.equals("planes")){
				numPlan = numPlan+1;
			}
			if(classVal.equals("moderate wind")){
				numMwind = numMwind+1;
			}
			if(classVal.equals("morning chorus")){
				numChorus = numChorus+1;
			}
			if(classVal.equals("strong wind")){
				numSwind = numSwind+1;
			}
			if(classVal.equals("loud cicadas")){
				numLcica = numLcica+1;
			}

		}

		//------------Analysing the selected sequences & their classes-----------------

				
		//**************************************** Seed Set ****************************************************//

		
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++SIMILARITY-BASED AL++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		
		// After selecting a seed set, for similarity-based approaches (IDiv, IDen, IDD approaches), 
		//we need to divide the data into  subsets using stratified sampling to reduce the similarity computations.
		//Then we divide the data into some folds (parts)and run similarity-based AL approaches for each part.
		
	
		//+++++++++++++++ Stratified Subsampling Data +++++++++++++++++++++++//
		//subsampling setting
		
		//Now we need to make a loop of 40 (# of folds) and in each loop get one of the folds and apply the AL approach on the whole fold:
		int NoFolds = 40; //each fold should contain around 10,405 samples
		for (int f=1; f<=NoFolds; f++) { 
			
			StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
			srf.setNumFolds(NoFolds);
			srf.setInputFormat(allData); // allData.size = 416236 after seed selection
			srf.setSeed(2);
			srf.setFold(f);
			
			Instances sampledInsts = Filter.useFilter(allData, srf);
			
			// write the current fold into a file
			BufferedWriter subsetwriter = new BufferedWriter(new FileWriter(resultsDirectory +"/" + "Fold"+ f+".arff"));
			subsetwriter.write(sampledInsts.toString());
			subsetwriter.flush();
			subsetwriter.close();
			
			//------------calculating densities----------
			CalDensities densitycalculator = new CalDensities();
			HashMap densityhash = densitycalculator.computeDensities(sampledInsts);
			//------------calculating densities----------
			
			int stopPoint = sampledInsts.size()/bunchSize;   //  Continue AL loop until reaching the end of ULn
			
			for (int batchCounter = 1 ; batchCounter <= stopPoint ; batchCounter++){    // active learning loop
				
				Classifier clsAL = new RandomForest(); //SMO(); //J48();//
				clsAL.buildClassifier(Ln);

				
				//+++++++++++++++ Evaluation+++++++++++++++++++++++//
				Evaluation evaluator2 = new Evaluation(testData);
				Classifier ALmodel = clsAL;  	
				bw.write("Active(Bunch " + (Ln.size()) + " Testset size: " + testData.size() + ")\n");
				bw.write("Number of Insects = " + numIns + " Numebr of Birds = " + numBird + " Numebr of Fairly quiet = " + 
						numFquiet + " Numebr of Very quiet = " + numVquiet + " Numebr of Light wind = " + numLwind +
						" Numebr of Light rain = " + numLrain + " Numebr of Cicadas = " + numCica + 
						" Numebr of Moderate rain = " + numMrain + " Numebr of Planes = " + numPlan + 
						" Numebr of Moderate wind = " + numMwind + " Numebr of Morning Chorus = " + numChorus + 
						" Numebr of Strong wind = " + numSwind + " Numebr of Loud Cicadas = " + numLcica + "\n");
				evaluator2.evaluateModel(ALmodel, testData);
				bw.write(evaluator2.toSummaryString("\nResults\n======\n", true));
				bw.write(evaluator2.toClassDetailsString("\nResults Per Class\n======\n"));
				bw.write(evaluator2.toMatrixString("\nConfusion Matrix\n======\n"));  //output confusion matrix
				bw.write("-------------------------------------------------------\n\n");
				bw.flush();
				
				//Due to high number of models generated, We only save those with performance higher than a threshold.
				// only models with Accuracy>87 will be saved
				String s = evaluator2.toSummaryString();
				String[] arr = s.split(" +"); 
				if (Double.parseDouble(arr[4])>90.9) {
					ObjectOutputStream sAL =
							new ObjectOutputStream(new FileOutputStream(resultsDirectory +"/"+ "ALmodel-" + Ln.size() + ".model"));
					sAL.writeObject(clsAL);
					sAL.close();
				}
				
				//------------calculating diversities----------
//				CalDiversities diversitycalculator = new CalDiversities();
//				HashMap diversityhash = diversitycalculator.computeDiversities(Ln ,sampledInsts);
				//------------calculating diversities----------
				
				
				
				// +++++++++++++++++++++ Select instances by using Information Density algorithm
				int[] selectedInstanceIndex = new int [bunchSize];
				InformationDensityQuerier queryDens = new InformationDensityQuerier(densityhash) {
				};
				selectedInstanceIndex = queryDens.select((Classifier) ALmodel, sampledInsts, Ln, bunchSize);
//									
			   // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
									
									
				// +++++++++++++++++++++ Select instances by using Information Diversity algorithm
//				int[] selectedInstanceIndex = new int [bunchSize];
//				InformationDiversityQuerier queryDiv = new InformationDiversityQuerier(diversityhash) {
//				};
//				selectedInstanceIndex = queryDiv.select((Classifier) ALmodel, sampledInsts, Ln, bunchSize);
									
				// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

									
				// +++++++++++++++++++++ Select instances by using IDD algorithm
//				int[] selectedInstanceIndex = new int [bunchSize];
//				IDDQuerier queryDD = new IDDQuerier(densityhash, diversityhash) {
//				};
//				selectedInstanceIndex = queryDD.select((Classifier) ALmodel, sampledInsts, Ln, bunchSize);
				//			
				//// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	
				
				//------------Analysing the selected sequences & their classes-----------------
				
				for(int instCount= selectedInstanceIndex.length-1; instCount >=0  ; instCount--){  
					
					Instance inst = sampledInsts.get(selectedInstanceIndex[instCount]);
					String classVal = inst.stringValue(inst.classAttribute());
					if(classVal.equals("insects")){
						numIns = numIns+1;
					}
					if(classVal.equals("birds")){
						numBird = numBird+1;
					}
					if(classVal.equals("fairly quiet")){
						numFquiet = numFquiet+1;
					}
					if(classVal.equals("very quiet")){
						numVquiet = numVquiet+1;
					}
					if(classVal.equals("light wind")){
						numLwind = numLwind+1;
					}
					if(classVal.equals("light rain")){
						numLrain = numLrain+1;
					}
					if(classVal.equals("cicadas")){
						numCica = numCica+1;
					}
					if(classVal.equals("moderate rain")){
						numMrain = numMrain+1;
					}
					if(classVal.equals("planes")){
						numPlan = numPlan+1;
					}
					if(classVal.equals("moderate wind")){
						numMwind = numMwind+1;
					}
					if(classVal.equals("morning chorus")){
						numChorus = numChorus+1;
					}
					if(classVal.equals("strong wind")){
						numSwind = numSwind+1;
					}
					if(classVal.equals("loud cicadas")){
						numLcica = numLcica+1;
					}
					
					try{                 //++++++++++++++++++++++++for only AL
						
						Ln.add(sampledInsts.remove(selectedInstanceIndex[instCount]));

					}catch(IndexOutOfBoundsException e){
						System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + sampledInsts.size());
						//System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + resampledI.size()); //****************************** ADDED for IDen, IDiv, IDD
					}                    //++++++++++++++++++++++++for only AL
				}
				System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + sampledInsts.size());
				
				//------------Analysing the selected sequences & their classes-----------------
				
			} //end of active learning loop 
			
			//add the remaining samples of a fold to the labeled set and build a final model
//			if(sampledInsts.size()>0){  //add the remaining samples of a fold to the labeled set 
//
//				while(sampledInsts.size()>=1){
//					Ln.add(sampledInsts.remove(0));
//				}
//			}
//
//			System.out.println("-------------------------------------------------------------------------------------------");
//			System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + sampledInsts.size());
			//add the remaining samples of a fold to the labeled set
			
			
//			Classifier ALmodelFinal = new RandomForest(); // J48();//SMO(); //
//			ALmodelFinal.buildClassifier(Ln);
//			ObjectOutputStream sAL =
//					new ObjectOutputStream(new FileOutputStream(resultsDirectory +"/"+ "ALmodel-" + Ln.size() + ".model")); // (resultsDirectory + "ALmodel-" + Ln.size() + ".model"));
//			sAL.writeObject(ALmodelFinal);
//			sAL.close();
//			
//			//+++++++++++++++ Evaluation+++++++++++++++++++++++//
//			Evaluation evaluator3 = new Evaluation(testData);
//			Classifier ALmodel = ALmodelFinal; //Classifier ALmodel = _allearner.getModel();   //		
//			bw.write("Active(Bunch " + (Ln.size()) + " Testset size: " + testData.size() + ")\n");
//			evaluator3.evaluateModel(ALmodel, testData);
//			bw.write(evaluator3.toSummaryString("\nResults\n======\n", true));
//			bw.write(evaluator3.toClassDetailsString("\nResults Per Class\n======\n"));
//			bw.write(evaluator3.toMatrixString("\nConfusion Matrix\n======\n"));  //output confusion matrix
//			bw.write("-------------------------------------------------------\n\n");
//			bw.flush();
			
			//add the remaining samples of a fold to the labeled set and build a final model
			
			
		} // end of processing one fold of the data
		
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++SIMILARITY-BASED AL++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
//++++++++++++++++++++++++++++++++++++++++++++++NON-SIMILARITY-BASED AL APPROACHES+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		
		
//		int stopPoint = allData.size()/bunchSize;   //  Continue AL loop until reaching the end of ULn
//		for (int batchCounter = 1 ; batchCounter <= stopPoint ; batchCounter++){    // active learning loop    
//			
//			
//			Classifier clsAL = new RandomForest(); //J48();//SMO(); //
//			clsAL.buildClassifier(Ln);
//
//			//+++++++++++++++ Evaluation+++++++++++++++++++++++//
//			Evaluation evaluator2 = new Evaluation(testData);
//			Classifier ALmodel = clsAL;  
//			bw.write("Active(Bunch " + batchCounter + " Labelled set size: " + (Ln.size()) + " Testset size: " + testData.size() + ")\n");
//			bw.write("AL: Number of Insects = " + numIns + " Numebr of Birds = " + numBird + " Numebr of Fairly quiet = " + 
//					numFquiet + " Numebr of Very quiet = " + numVquiet + " Numebr of Light wind = " + numLwind +
//					" Numebr of Light rain = " + numLrain + " Numebr of Cicadas = " + numCica + 
//					" Numebr of Moderate rain = " + numMrain + " Numebr of Planes = " + numPlan + 
//					" Numebr of Moderate wind = " + numMwind + " Numebr of Morning Chorus = " + numChorus + 
//					" Numebr of Strong wind = " + numSwind + " Numebr of Loud Cicadas = " + numLcica + "\n");
//			evaluator2.evaluateModel(ALmodel, testData);
//			bw.write(evaluator2.toSummaryString("\nResults\n======\n", true));
//			//System.out.println(evaluator.toSummaryString("\nResults\n======\n", true));
//			bw.write(evaluator2.toClassDetailsString("\nResults Per Class\n======\n"));
//			bw.write(evaluator2.toMatrixString("\nConfusion Matrix\n======\n"));  //output confusion matrix
//			bw.write("-------------------------------------------------------\n\n");
//			bw.flush();
//			//+++++++++++++++ Evaluation+++++++++++++++++++++++//
//			
//			
//			// Due to high number of models generated, We only save those with performance higher than a threshold.
//			// only models with Accuracy>88 will be saved
////			String s = evaluator2.toSummaryString();
////			String[] arr = s.split(" +"); 
////			if (Double.parseDouble(arr[4])>89.9) {
////				ObjectOutputStream sAL =
////						new ObjectOutputStream(new FileOutputStream(resultsDirectory +"/"+ "ALmodel-" + Ln.size() + ".model"));
////				sAL.writeObject(clsAL);
////				sAL.close();
////			}
//			
//			
//			//+++++++++++++++++++++ Select instances by using Least Conf algorithm
////			int[] selectedInstanceIndex = new int [bunchSize];
////			LeastConfQuerier queryLC = new LeastConfQuerier() {};
////			selectedInstanceIndex = queryLC.select((Classifier) ALmodel, allData, Ln, bunchSize);
////			//				//		
//			//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//						
//			//+++++++++++++++++++++ Select instances by using Random Sampling algorithm
////			int[] selectedInstanceIndex = new int [bunchSize];
////			RandomQuerier queryRS = new RandomQuerier() {
////			};
////			selectedInstanceIndex = queryRS.select((Classifier) ALmodel, allData, Ln, bunchSize);
//			//				//		
//			//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//					
//			
//			// +++++++++++++++++++++ Select instances by using Entropy algorithm
////			int[] selectedInstanceIndex = new int [bunchSize];
////			EntropyQuerier queryE = new EntropyQuerier() {   // Sequence Entropy
////			};
////			selectedInstanceIndex = queryE.select((Classifier) ALmodel, allData, Ln, bunchSize);
//			
////			+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//			
//			
//			// +++++++++++++++++++++ Select instances by using Margin algorithm
////			int[] selectedInstanceIndex = new int [bunchSize];
////			MarginQuerier queryMg = new MarginQuerier() {};
////			selectedInstanceIndex = queryMg.select((Classifier) ALmodel, allData, Ln, bunchSize);  
//						
//			// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//			
////			for(int instCount= selectedInstanceIndex.length-1; instCount >=0  ; instCount--){   
////				try{
////					
////					Ln.add(allData.remove(selectedInstanceIndex[instCount]));
////				}catch(IndexOutOfBoundsException e){
////					System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + allData.size());
////				}
////			}
//			
//			//------------Analysing the selected sequences & their classes-----------------
//
//			for(int instCount= selectedInstanceIndex.length-1; instCount >=0  ; instCount--){  
//				Instance inst = allData.get(selectedInstanceIndex[instCount]);
//				String classVal = inst.stringValue(inst.classAttribute());
//				if(classVal.equals("insects")){
//					numIns = numIns+1;
//				}
//				if(classVal.equals("birds")){
//					numBird = numBird+1;
//				}
//				if(classVal.equals("fairly quiet")){
//					numFquiet = numFquiet+1;
//				}
//				if(classVal.equals("very quiet")){
//					numVquiet = numVquiet+1;
//				}
//				if(classVal.equals("light wind")){
//					numLwind = numLwind+1;
//				}
//				if(classVal.equals("light rain")){
//					numLrain = numLrain+1;
//				}
//				if(classVal.equals("cicadas")){
//					numCica = numCica+1;
//				}
//				if(classVal.equals("moderate rain")){
//					numMrain = numMrain+1;
//				}
//				if(classVal.equals("planes")){
//					numPlan = numPlan+1;
//				}
//				if(classVal.equals("moderate wind")){
//					numMwind = numMwind+1;
//				}
//				if(classVal.equals("morning chorus")){
//					numChorus = numChorus+1;
//				}
//				if(classVal.equals("strong wind")){
//					numSwind = numSwind+1;
//				}
//				if(classVal.equals("loud cicadas")){
//					numLcica = numLcica+1;
//				}
//
////				try{                 //++++++++++++++++++++++++for only AL
////					
////					Ln.add(allData.remove(selectedInstanceIndex[instCount]));
////					//Ln.add(resampledI.remove(selectedInstanceIndex[instCount])); //****************************** ADDED for IDen, IDiv, IDD
////				}catch(IndexOutOfBoundsException e){
////					System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + allData.size());
////					//System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + resampledI.size()); //****************************** ADDED for IDen, IDiv, IDD
////				}                    //++++++++++++++++++++++++for only AL
//			}
//			System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + allData.size());
//			//System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + resampledI.size());//****************************** ADDED for IDen, IDiv, IDD
////			
//			//------------Analysing the selected sequences & their classes-----------------
//			
//			//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//			for(int instCount= selectedInstanceIndex.length-1; instCount >=0  ; instCount--){   // Add instances selected in AL to Ln 
//				Ln.add(allData.get(selectedInstanceIndex[instCount]));
//			}
//			
//			ArrayList<Integer> allSelectedIndex = new ArrayList<Integer>();
//			for (int i= 0; i<selectedInstanceIndex.length; i++){  
//				allSelectedIndex.add(selectedInstanceIndex[i]);
//			}
//
//			Collections.sort(allSelectedIndex);
//			
//
//			//+++++++++++++++++++remove instances selected in AL
//			for (int i=allSelectedIndex.size()-1; i>=0; i-- ){     
//				try{
//					allData.remove(allSelectedIndex.get(i));  
//					}catch(IndexOutOfBoundsException e){
//						System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + allData.size());
//					}
//			
//			}
//			//+++++++++++++++++++remove instances selected in AL 
//
//		} //end of active learning loop  
//
//		if(allData.size()>0){
//
//			while(allData.size()>=1){
//				Ln.add(allData.remove(0));
//			}
//		}
//
//		System.out.println("-------------------------------------------------------------------------------------------");
//		System.out.println("---------------------------------LnSize=" + Ln.size() + "--ULnSize=" + allData.size());
//		
//		
//		
//		Classifier ALmodelFinal = new RandomForest(); //J48();// SMO(); //
//		ALmodelFinal.buildClassifier(Ln);
//		
//		
//		//+++++++++++++++ Evaluation+++++++++++++++++++++++//
//		Evaluation evaluator3 = new Evaluation(testData);
//		Classifier ALmodel = ALmodelFinal; 	
//		bw.write("Active(Bunch " + (Ln.size()) + " Testset size: " + testData.size() + ")\n");
//		bw.write("Number of Insects = " + numIns + " Numebr of Birds = " + numBird + " Numebr of Fairly quiet = " + 
//				numFquiet + " Numebr of Very quiet = " + numVquiet + " Numebr of Light wind = " + numLwind +
//				" Numebr of Light rain = " + numLrain + " Numebr of Cicadas = " + numCica + 
//				" Numebr of Moderate rain = " + numMrain + " Numebr of Planes = " + numPlan + 
//				" Numebr of Moderate wind = " + numMwind + " Numebr of Morning Chorus = " + numChorus + 
//				" Numebr of Strong wind = " + numSwind + " Numebr of Loud Cicadas = " + numLcica + "\n");
//		evaluator3.evaluateModel(ALmodel, testData);
//		bw.write(evaluator3.toSummaryString("\nResults\n======\n", true));
//		bw.write(evaluator3.toClassDetailsString("\nResults Per Class\n======\n"));
//		bw.write(evaluator3.toMatrixString("\nConfusion Matrix\n======\n"));  //output confusion matrix
//		bw.write("-------------------------------------------------------\n\n");
//		bw.flush();
//		//+++++++++++++++ Evaluation+++++++++++++++++++++++//
//		
//		// Due to high number of models generated, We only save those with performance higher than a threshold.
//		// only models with Accuracy>88 will be saved
////		String s = evaluator3.toSummaryString();
////		String[] arr = s.split(" +"); 
////		if (Double.parseDouble(arr[4])>89.9) {
////			ObjectOutputStream sAL =
////					new ObjectOutputStream(new FileOutputStream(resultsDirectory +"/"+ "ALmodel-" + Ln.size() + ".model"));
////			sAL.writeObject(ALmodelFinal);
////			sAL.close();
////		}
		
//++++++++++++++++++++++++++++++++++++++++++++++NON-SIMILARITY-BASED AL APPROACHES+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		

	}  // ****************************************END OF Learner over whole Train**************************************************

  
}