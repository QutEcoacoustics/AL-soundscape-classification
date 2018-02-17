
public class ALClassifier {

	
	
	public void ALClassifier()
	{
	}

	public static void main(String[] args) throws Exception
	{
	

		String[] myOptions= new String[5];
		myOptions[0]= "-train";
		myOptions[1]= "train_data.csv";
		myOptions[2]= "-test";
		myOptions[3]= "test";
		myOptions[4]= "-evaluate";

		System.out.println(Runtime.getRuntime().maxMemory()+"   "+ Runtime.getRuntime().freeMemory()+"   " + Runtime.getRuntime().totalMemory()); 
		Classifier.main(myOptions);

	}
	
}
