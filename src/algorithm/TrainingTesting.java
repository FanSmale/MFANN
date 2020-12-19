package algorithm;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;

/**
 * Training and testing using ANN.
 * 
 * @author Fan Min. minfanphd@163.com.
 *
 */
public class TrainingTesting {
	/**
	 * The neural network.
	 */
	BPNeuralNetwork network;

	/**
	 * The input of the whole data (D), only conditions.
	 */
	double[][] wholeDataInput;

	/**
	 * The input of the whole data (D), only conditions.
	 */
	double[][] wholeDataOutput;

	/**
	 * Number of conditions.
	 */
	int numConditions;

	/**
	 * Number of classes, 2 for binary classification. For classification.
	 */
	int numClasses;

	/**
	 * The training data input (X).
	 */
	double[][] trainingDataInput;

	/**
	 * The training data input (Y).
	 */
	double[][] trainingDataOutput;

	/**
	 * The testing data input (X').
	 */
	double[][] testingDataInput;

	/**
	 * The training data input (Y').
	 */
	double[][] testingDataOutput;

	/**
	 * For random.
	 */
	static Random random = new Random();

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The arff filename
	 ********************
	 */
	public TrainingTesting(String paraFilename) {
		Instances tempInstances = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			tempInstances = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n"
					+ ee);
			System.exit(0);
		} // Of try

		numConditions = tempInstances.numAttributes() - 1;
		numClasses = tempInstances.attribute(numConditions).numValues();

		wholeDataInput = new double[tempInstances.numInstances()][numConditions];
		for (int i = 0; i < wholeDataInput.length; i++) {
			for (int j = 0; j < wholeDataInput[0].length; j++) {
				wholeDataInput[i][j] = tempInstances.instance(i).value(j);
			}// Of for j
		}// Of for i

		// Scale the decision to an array.
		wholeDataOutput = new double[tempInstances.numInstances()][numClasses];
		for (int i = 0; i < wholeDataOutput.length; i++) {
			int tempDecision = (int) tempInstances.instance(i).value(
					numConditions);
			wholeDataOutput[i][tempDecision] = 1;
		}// Of for i
	}// Of the constructor

	/**
	 ********************
	 * Split the whole data into the training and testing parts.
	 * 
	 * @param paraFraction
	 *            The fraction of the training set.
	 ********************
	 */
	public void splitInTwo(double paraFraction) {
		// Step 1. Allocate pesudo space.
		int tempTrainingSize = (int) (wholeDataInput.length * paraFraction);
		int tempTestingSize = wholeDataInput.length - tempTrainingSize;

		trainingDataInput = new double[tempTrainingSize][];
		trainingDataOutput = new double[tempTrainingSize][];

		testingDataInput = new double[tempTestingSize][];
		testingDataOutput = new double[tempTestingSize][];

		// Step 2. Determine the parts.
		int[] tempIndices = getRandomOrder(wholeDataInput.length);

		// Step 3. Copy data.
		for (int i = 0; i < tempTrainingSize; i++) {
			trainingDataInput[i] = wholeDataInput[tempIndices[i]];
			trainingDataOutput[i] = wholeDataOutput[tempIndices[i]];
		}// Of for i

		for (int i = 0; i < tempTestingSize; i++) {
			testingDataInput[i] = wholeDataInput[tempIndices[tempTrainingSize
					+ i]];
			testingDataOutput[i] = wholeDataOutput[tempIndices[tempTrainingSize
					+ i]];
		}// Of for i
	}// Of splitInTwo

	/**
	 ********************************** 
	 * Get a random order index array.
	 * 
	 * @param paraLength
	 *            The length of the array.
	 * @return A random order.
	 ********************************** 
	 */
	public static int[] getRandomOrder(int paraLength) {
		// Step 1. Initialize
		int[] resultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			resultArray[i] = i;
		} // Of for i

		// Step 2. Swap many times
		int tempFirst, tempSecond;
		int tempValue;
		for (int i = 0; i < paraLength * 10; i++) {
			tempFirst = random.nextInt(paraLength);
			tempSecond = random.nextInt(paraLength);

			tempValue = resultArray[tempFirst];
			resultArray[tempFirst] = resultArray[tempSecond];
			resultArray[tempSecond] = tempValue;
		} // Of for i

		return resultArray;
	}// Of getRandomOrder

	/**
	 ********************
	 * Train the network
	 * 
	 * @param paraLayerNumNodes
	 *            The number of nodes for each layer (may be different).
	 * @param paraRate
	 *            Learning rate.
	 * @param paraMobp
	 *            动量系数
	 * @param paraRounds
	 *            The training rounds.
	 ********************
	 */
	public void train(int[] paraLayerNumNodes, double paraRate,
			double paraMobp, double paraRounds) {
		paraLayerNumNodes[0] = numConditions;
		paraLayerNumNodes[paraLayerNumNodes.length - 1] = numClasses;
		network = new BPNeuralNetwork(paraLayerNumNodes, paraRate, paraMobp);

		for (int i = 0; i < paraRounds; i++) {
			for (int j = 0; j < trainingDataInput.length; j++) {
				// System.out.println("Training with " +
				// Arrays.toString(trainingDataInput[j])
				// + " and " + Arrays.toString(trainingDataOutput[j]));
				network.train(trainingDataInput[j], trainingDataOutput[j]);
			}// Of for j
		}// Of for i
	}// Of train

	/**
	 ********************
	 * test the network with the testing set.
	 * 
	 * @return The accuracy.
	 ********************
	 */
	public double test() {
		double tempCorrect = 0;
		double[] tempOut;
		int tempMaxIndex;
		double tempMax;
		for (int i = 0; i < testingDataInput.length; i++) {
			tempOut = network.computeOut(testingDataInput[i]);
			// System.out.println("The prediction is: " +
			// Arrays.toString(tempOut)
			// + " while the actual output is: " +
			// Arrays.toString(testingDataOutput[i]));
			tempMax = tempOut[0];
			tempMaxIndex = 0;

			for (int j = 1; j < tempOut.length; j++) {
				if (tempMax < tempOut[j]) {
					tempMax = tempOut[j];
					tempMaxIndex = j;
				}// Of if
			}// Of for j

			if (testingDataOutput[i][tempMaxIndex] == 1) {
				tempCorrect++;
			}// Of if
		}// Of for i
		System.out.println("" + tempCorrect + " correct among "
				+ testingDataInput.length + " instances.");
		return tempCorrect / testingDataInput.length;
	}// Of test

	/**
	 ********************
	 * Show me.
	 ********************
	 */
	public String toString() {
		String resultString = "";
		resultString += "The whole data input is: \r\n"
				+ Arrays.deepToString(wholeDataInput);
		resultString += "\r\nThe whole data output is: \r\n"
				+ Arrays.deepToString(wholeDataOutput);
		resultString += "\r\nThe training data intput is: \r\n"
				+ Arrays.deepToString(trainingDataInput);
		resultString += "\r\nThe training data output is: \r\n"
				+ Arrays.deepToString(trainingDataOutput);
		resultString += "\r\nThe testing data intput is: \r\n"
				+ Arrays.deepToString(testingDataInput);
		resultString += "\r\nThe testing data output is: \r\n"
				+ Arrays.deepToString(testingDataOutput);

		return resultString;
	}// Of toString

	/**
	 ********************
	 * Unit test.
	 ********************
	 */
	public static void main(String[] args) {
		TrainingTesting tempTrainingTesting = new TrainingTesting(
		// "src/data/iris.arff");
				"src/data/wdbc_norm_ex.arff");
		tempTrainingTesting.splitInTwo(0.7);
		// System.out.println(tempTrainingTesting);

		tempTrainingTesting.train(new int[] { 4, 10, 8, 8, 3 }, 0.15, 0.8, 10);
		double tempAccuracy = tempTrainingTesting.test();

		System.out.println("The accuracy is: " + tempAccuracy);
	}// Of main
}// Of class TrainingTesting
