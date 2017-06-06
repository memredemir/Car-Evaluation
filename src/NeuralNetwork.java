/**
 * Ilk basta neural networkumuzu layerlarimizin boyutuna gore olusturuyoruz.
 * run metoduna maxstep sayisini ve input arrayini gonderiyoruz.
 * run metodunda inputlari set edip activate metodunu cagiriyoruz.
 * activate metodu ise bize output hesaplayip degerini donuyor.
 * bu dondugu degeri resultoutputs arrayine atiyoruz.
 * erroru de farklarin kareleri toplami yontemi ile hesapliyoruz.
 * backpropogationa expectedoutputu gonderip error check yapiyoruz.
 * Eger error minerrore ulasirsa train sonlaniyor.
 * Sonra train degerlerini bastiriyoruz.
 * Test metoduna test inputlarini gonderiyoruz.
 * Testimizi yapip sonuclari bastiriyoruz.
 * @author emre & atacan
 */


import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.*;
import java.util.*;
 


//datasette good ve v_good olanlari artir.
public class NeuralNetwork
{
    final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
 
    final boolean isTrained = false;
    final DecimalFormat df;
    
    final Random rand = new Random();
    final Neuron bias = new Neuron();
    
    final int[] layers;
    final int randomWeightMultiplier = 1;
 
    final double epsilon = 0.00000000001;
    final double learningRate = 0.9f;
    final double momentum = 0.7f;
    
    final static double inputs[][] = new double[1868][21];
    final static double input_test[][] = new double[328][21];
    final double expectedOutputs[][] = output("CarEvaluation.txt"); 
    
    double resultOutputs[][] = new double[inputs.length][4]; 
    double output[];
 
    // for weight update all
    final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();
 
    //Main method
    public static void main(String[] args) 
    {
    	input("CarEvaluation.txt");
        NeuralNetwork nn = new NeuralNetwork(21, 4, 4);
        int maxRuns = 1000;
        double minErrorCondition = 0.001;
        nn.run(maxRuns, minErrorCondition,inputs);
        
        nn.Test(input_test);
    }
     //Data setinin test edildigi method.
     public void Test(double [][] input){
    	double result[][] = new double[input.length][4];
    	double dogru_ua = 0,dogru_a = 0, dogru_g = 0, dogru_vg = 0, yanlis_ua = 0, yanlis_a = 0, yanlis_g = 0, yanlis_vg = 0;
    	
    	for(int i=0; i<input.length;i++){
        	setInput(input[i]);
        	activate();
        	double output[] = getOutput();
        	result[i]=output;	
        }
    	
    	for(int i = 0; i < result.length;i++){
    		int max = maxCal(result[i]); 
    		int max1 = maxCal(expectedOutputs[i+1868]);
    		
    		if(max == max1)
    		{
    			if(result[i][max] > 0.57)
    			{
    				switch(max){
    				
    				case 0:
    					dogru_ua++;
    					break;
    				case 1:
    					dogru_a++;
    					break;
    				case 2:
    					dogru_g++;
    					break;
    				case 3:
    					dogru_vg++;
    					break;
    				}
    			}
    			else
    				switch(max){
    				
	    				case 0:
	    					yanlis_ua++;
	    					break;
	    				case 1:
	    					yanlis_a++;
	    					break;
	    				case 2:
	    					yanlis_g++;
	    					break;
	    				case 3:
	    					yanlis_vg++;
	    					break;
    				}
    		}
    		else
    			switch(max1){
				
				case 0:
					yanlis_ua++;
					break;
				case 1:
					yanlis_a++;
					break;
				case 2:
					yanlis_g++;
					break;
				case 3:
					yanlis_vg++;
					break;
			}
    		
    	}
    	
    	for(int p=0; p<input.length; p++){
		    	System.out.print("INPUTS: ");
	            for (int x = 0; x < layers[0]; x++) {
	                System.out.print(input[p][x] + " ");
	            }
	         
		    	System.out.print("Outputs: ");
	            for (int x = 0; x < layers[2]; x++) {
	                System.out.print(result[p][x] + " ");
	            }
	           
	    	 System.out.println("");
    	}
    	
    	double[] dogrular = {dogru_ua,dogru_a,dogru_g,dogru_vg};
    	double[] yanlislar = {yanlis_ua,yanlis_a,yanlis_g,yanlis_vg};
    	
    	for(int i = 0;i<4;i++){
    		System.out.println(i+". output indexi icin %"+100 * (dogrular[i]/(dogrular[i]+yanlislar[i])) + " dogru tahmin!");   	
    	}
    	
    	System.out.println("Toplam dogruluk orani : %" + (dogrular[0]+dogrular[1]+dogrular[2]+dogrular[3])/328 * 100);
    }
 

     /**
      * Constructor.
      * Butun noronlari ve connectionlari neuron classinida kullanarak olusturuyor.
      */
    public NeuralNetwork(int input, int hidden, int output) {
        this.layers = new int[] { input, hidden, output };
        df = new DecimalFormat("#.0#");
 
        for (int i = 0; i < layers.length; i++) {
            if (i == 0) { // input layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    inputLayer.add(neuron);
                }
            } else if (i == 1) { // hidden layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(inputLayer);
                    neuron.addBiasConnection(bias);
                    hiddenLayer.add(neuron);
                }
            }
 
            else if (i == 2) { // output layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(hiddenLayer);
                    neuron.addBiasConnection(bias);
                    outputLayer.add(neuron);
                }
            } else {
                System.out.println("!Error NeuralNetwork init");
            }
        }
 
        // initialize random weights
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
 
        // reset id counters
        Neuron.counter = 0;
        Connection.counter = 0;
 
        if (isTrained) {
            trainedWeights();
            updateAllWeights();
        }
    }
 
    // random sayi uretmek
    double getRandom() {
        return randomWeightMultiplier * (rand.nextDouble() * 2 - 1); // [-1;1[
    }
 
    /**
     * inputlayerin degerlerini atamak icin.
     */
    public void setInput(double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }
 
    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }
 
    /**
     * neuron classi icinde output hesaplama.
     */
    public void activate() {
        for (Neuron n : hiddenLayer)
            n.calculateOutput();
        for (Neuron n : outputLayer)
            n.calculateOutput();
    }
 
    
    /**
     * Erroru partial derivative kullanarak hesapliyor.
     * Bias weight delta guncellemesi yapiyor.
     * expectedoutput degerini 0 1 aralgina normalize ediyor.
     */
    public void applyBackpropagation(double expectedOutput[]) {
 
        // error check, normalize value 0;1
        for (int i = 0; i < expectedOutput.length; i++) {
            double d = expectedOutput[i];
            if (d < 0 || d > 1) {
                if (d < 0)
                    expectedOutput[i] = 0 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        }
 
        int i = 0;
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double ak = n.getOutput();
                double ai = con.leftNeuron.getOutput();
                double desiredOutput = expectedOutput[i];
 
                double partialDerivative = -ak * (1 - ak) * ai
                        * (desiredOutput - ak);
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
            i++;
        }
 
        // update weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                double aj = n.getOutput();
                double ai = con.leftNeuron.getOutput();
                double sumKoutputs = 0;
                int j = 0;
                for (Neuron out_neu : outputLayer) {
                    double wjk = out_neu.getConnection(n.id).getWeight();
                    double desiredOutput = (double) expectedOutput[j];
                    double ak = out_neu.getOutput();
                    j++;
                    sumKoutputs = sumKoutputs
                            + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
                }
 
                double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
            }
        }
    }
 
    
    
   /** run metoduna maxstep sayisini ve input arrayini gonderiyoruz.
    * run metodunda inputlari set edip activate metodunu cagiriyoruz.
    * activate metodu ise bize output hesaplayip degerini donuyor.
    * bu dondugu degeri resultoutputs arrayine atiyoruz.
    * erroru de farklarin kareleri toplami yontemi ile hesapliyoruz.
    * backpropogationa expectedoutputu gonderip error check yapiyoruz.
    * Eger error minerrore ulasirsa train sonlaniyor.
    * Sonra train degerlerini bastiriyoruz.
    */
    public void run(int maxSteps, double minError,double [][] inputs) {
        int i;
        // Train neural network minErrore ulasana kadar yada maxSteps asilincaya kadar.
        double error = 1;
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            for (int p = 0; p < inputs.length; p++) {
                setInput(inputs[p]);
 
                activate();
 
                output = getOutput();
                resultOutputs[p] = output;
 
                for (int j = 0; j < expectedOutputs[p].length; j++) {
                    double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
                    error += err;
                }
 
                applyBackpropagation(expectedOutputs[p]);
            }
        }
 
        printResult();
         
        System.out.println("Sum of squared errors = " + error);
    }
     
    public void printResult()
    {
        System.out.println("NN example with xor training");
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("INPUTS: ");
            for (int x = 0; x < layers[0]; x++) {
                System.out.print(inputs[p][x] + " ");
            }
 
            System.out.print("EXPECTED: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(expectedOutputs[p][x] + " ");
            }
 
            System.out.print("ACTUAL: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(resultOutputs[p][x] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }
 
    String weightKey(int neuronId, int conId) {
        return "N" + neuronId + "_C" + conId;
    }
 
    /**
     * Take from hash table and put into all weights
     */
    public void updateAllWeights() {
        // update weights for the output layer
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                String key = weightKey(n.id, con.id);
                double newWeight = weightUpdate.get(key);
                con.setWeight(newWeight);
            }
        }
        // update weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            for (Connection con : connections) {
                String key = weightKey(n.id, con.id);
                double newWeight = weightUpdate.get(key);
                con.setWeight(newWeight);
            }
        }
    }
 
    // trained data
    public void trainedWeights() {
        weightUpdate.clear();
         
        weightUpdate.put(weightKey(3, 0), 1.03);
        weightUpdate.put(weightKey(3, 1), 1.13);
        weightUpdate.put(weightKey(3, 2), -.97);
        weightUpdate.put(weightKey(4, 3), 7.24);
        weightUpdate.put(weightKey(4, 4), -3.71);
        weightUpdate.put(weightKey(4, 5), -.51);
        weightUpdate.put(weightKey(5, 6), -3.28);
        weightUpdate.put(weightKey(5, 7), 7.29);
        weightUpdate.put(weightKey(5, 8), -.05);
        weightUpdate.put(weightKey(6, 9), 5.86);
        weightUpdate.put(weightKey(6, 10), 6.03);
        weightUpdate.put(weightKey(6, 11), .71);
        weightUpdate.put(weightKey(7, 12), 2.19);
        weightUpdate.put(weightKey(7, 13), -8.82);
        weightUpdate.put(weightKey(7, 14), -8.84);
        weightUpdate.put(weightKey(7, 15), 11.81);
        weightUpdate.put(weightKey(7, 16), .44);
    }

    
    public static void input(String s){
    	
    	FileReader fr = null;
		try {
			fr = new FileReader(new File(s));
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        BufferedReader reader = new BufferedReader(fr);
        String line = "";
        String[] values;
        int j=0;
        try {
			while ((line = reader.readLine()) != null) {
				values = line.split(",");
				if(j<1868){
					for(int i=0; i<values.length-4;i++)
						inputs[j][i] = Double.parseDouble(values[i]);
					j++;
				}
				else{
					for(int i=0; i<values.length-4;i++)
						input_test[j-1868][i] = Double.parseDouble(values[i]);
					j++;
				}
			}
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	
    }
    
    public double[][] output(String s){
    	
    	double output[][] = new double[2196][4];
    	
    	FileReader fr = null;
		try {
			fr = new FileReader(new File(s));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        BufferedReader reader = new BufferedReader(fr);
        String line = "";
        String[] values;
        int j=0;
        try {
			while ((line = reader.readLine()) != null) {
				values = line.split(",");
				
				for(int i=values.length-4; i<values.length;i++)
					output[j][i-(values.length-4)] = Double.parseDouble(values[i]);
				j++;
			}
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	return output;
    	
    }
    
    public int maxCal(double[] arr){
    	int index = 0;
    	
    	for(int i = 0; i < arr.length; i++)
    		if(arr[i]>arr[index])	index = i;
    	
    	return index;
    	
    }
    
}