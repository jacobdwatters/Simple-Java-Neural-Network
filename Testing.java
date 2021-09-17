/**
 * Created Feb 12, 2019. Last Updated March 2, 2020
 * 
 * @author Jacob Watters
 */

import java.util.Random;
import java.text.DecimalFormat; 


/**
 * This class contains some test cases
 */
public class Testing {
	public static void main(String[] args) {
		 runTest1();
		// runTest2();
		// runTest3();
		// runTest4();
	}
	
	
	/**
	 * This test trains a network with 2 layers (1 hidden layer and an output)
	 */
	public static void runTest1() {
		int hiddenShape = 20, hiddenLayers = 1, inputShape = 2, outputShape = 1, batchSize = 4, epochs = 100000;	
		double learningRate = 0.1;
		double[][] samples = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		double[][] targets = { {0   }, {1   }, {1   }, {0   } };
		
		// Defines network
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(hiddenShape, hiddenLayers, inputShape, outputShape); 
		
		// Initializes parameters of network
		nn.compile(epochs, batchSize, learningRate);
		
		try {
			nn.train(samples, targets);
			
			System.out.println("\n\nTest 1:\n--------------------------------");
			System.out.println("Predictions");
			Matrix.print(nn.predict(samples[0]));
			Matrix.print(nn.predict(samples[1]));
			Matrix.print(nn.predict(samples[2]));
			Matrix.print(nn.predict(samples[3]));
			
			System.out.println("\n\nExpected");
			Matrix.print(targets);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * This test trains a network with 1 layer (an output)
	 */
	public static void runTest2 () {
		int inputShape = 2, outputShape = 1, batchSize = 4, epochs = 50000;	
		double learningRate = 0.1;
		double[][] samples = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		double[][] targets = { {0   }, {1   }, {1   }, {0   } };
		
		// Defines network
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(inputShape, outputShape); 
		// Initializes parameters of network
		nn.compile(epochs, batchSize, learningRate);
		
		try {
			nn.train(samples, targets);
			
			System.out.println("\n\nTest 2:\n--------------------------------");
			System.out.println("Predictions");
			Matrix.print(nn.predict(samples[0]));
			Matrix.print(nn.predict(samples[1]));
			Matrix.print(nn.predict(samples[2]));
			Matrix.print(nn.predict(samples[3]));
			
			System.out.println("\n\nExpected");
			Matrix.print(targets);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	/* This test trains on dataset where the first two values of a sample represent an and gate,
	 * the output of which is being OR'ed with the third value of the sample
	 */
	public static void runTest3() {
		// Note that if we round our final answers we could get a decent final loss with only 4000 epochs
		int hiddenShape = 20, hiddenLayers = 1, inputShape = 3, outputShape = 1, batchSize = 6, epochs = 30800;	
		double learningRate = 0.004; // Notice the lower learning rate here. It turns out 0.1 was too big a learning rate.
		double[][] samples = { {0, 0, 0	}, {0, 1, 0	}, {1, 0, 0	}, {1, 1, 0	}, {1, 1, 1	}, {0, 0, 1	} };
		double[][] targets = { {0   		}, {0   		}, {0   		}, {1   		}, {1		}, {1		} };
		
		// Defines network
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(hiddenShape, hiddenLayers, inputShape, outputShape); 
		
		// Initializes parameters of network
		nn.compile(epochs, batchSize, learningRate);
		
		try {
			nn.train(samples, targets);
			System.out.println("\n\nTest 1:\n--------------------------------");
			System.out.println("Predictions");
			Matrix.print(nn.predict(samples[0]));
			Matrix.print(nn.predict(samples[1]));
			Matrix.print(nn.predict(samples[2]));
			Matrix.print(nn.predict(samples[3]));
			Matrix.print(nn.predict(samples[4]));
			Matrix.print(nn.predict(samples[5]));
			
			System.out.println("\n\nExpected");
			Matrix.print(targets);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	// TODO: If the learning rate is too large here, the weights will explode and the final output will be NaN. Need to put in logic to
	// 		abort training if this happens? Maybe what is causing the NaN is it is becoming infinity
	//		then we are taking Infinity - Infinity? Possibly replace with BigDecimal?
	public static void runTest4() {
		int hiddenShape = 20, hiddenLayers = 3, inputShape = 3, outputShape = 1, batchSize = 8, epochs = 1500;	
		double learningRate = 0.5;
		double[][] samples = { {0, 0, 0	}, {0, 0, 1	}, {0, 1, 0	}, {0, 1, 1	}, {1, 0, 0	}, {1, 0, 1	}, {1, 1, 0	}, {1, 1, 1	} };
		double[][] targets = { {0		}, {1   		}, {0   		}, {1  		}, {0		}, {1		}, {1		}, {0		} };
		
		// Defines network
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(hiddenShape, hiddenLayers, inputShape, outputShape); 
		
		// Initializes parameters of network
		nn.compile(epochs, batchSize, learningRate);
		
		try {
			nn.train(samples, targets);
			System.out.println("\n\nTest 1:\n--------------------------------");
			System.out.println("Predictions");
			Matrix.print(nn.predict(samples[0]));
			Matrix.print(nn.predict(samples[1]));
			Matrix.print(nn.predict(samples[2]));
			Matrix.print(nn.predict(samples[3]));
			Matrix.print(nn.predict(samples[4]));
			Matrix.print(nn.predict(samples[5]));
			Matrix.print(nn.predict(samples[6]));
			Matrix.print(nn.predict(samples[7]));
			
			System.out.println("\n\nExpected");
			Matrix.print(targets);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}


/**
 * Allows for creation and training of a neural network with at most one hidden layer
 */
class SimpleNeuralNetwork {
	long seed = 1583128426457L;	/* seed for pseudo-random number
								 This seed has been specifically chosen to avoid getting stuck in
								 local minimums during gradient decent*/
	Random random = new Random(seed);
	
	boolean trained = false;
	
	private double learningRate 	= 0.1; 	// Learning rate of network
	private int epochs 			= 0;		// Iterations to train network
	private int batchSize		= 1;		// Number of samples to process before updating weights
	private int hiddenShape;				// Number of hidden layers
	
	private double[] 		lossPerEpoch;	/* A list of losses by epoch where the loss at a given index is the loss
											   for epoch index+1 i*/
	private double[][][] 	predictions;		// A list of model predictions by epoch
	private double[][][]		targetMatrix;	// Holds Target values converted to a list of matrices
	
	private double[][]		inputValues;		// Current input node Values
	private double[][][] 	hiddenValues; 	// Current hidden node values (Used when more than one hidden layer)
	private double[][] 		outputValues;	// Current output values
	
	private double[][] 		input2HiddenWeights;		// weights between inputs and hidden layer
	private double[][][]		hiddenWeights;			// weights between hidden Layers (currently not used)
	private double[][] 		hidden2OutputWeights;	// weights between hidden and output layer
	private double[][]		input2OutputWeights;		// weight updates between input and output layer (for single layer network)
	
	private double[][] 		deltaInput2HiddenWeights;	// weight updates between inputs and hidden layer
	private double[][][]		deltaHiddenWeights;			// weight updates between hidden Layers (currently not used)
	private double[][] 		deltaHidden2OutputWeights;	// weight updates between hidden and output layers
	private double[][]		deltaInput2OutputWeights;	// weight updates between input and output layer (for single layer network)
	
	
	/**
	 * Declares a network with the given shapes
	 * 
	 * @param hiddenShape - number of nodes in layer
	 * @param hiddenLayers - number of hiddenLayers (Currently only supports 0 or 1)
	 * @param inputShape - number of nodes in input layer
	 * @param outputShape - number of nodes in output layer
	 */
	public SimpleNeuralNetwork(int hiddenShape, int hiddenLayers, int inputShape, int outputShape) {
		this.inputValues		= new double[inputShape	][1];
		this.outputValues 	= new double[outputShape][1];
		this.hiddenShape = hiddenShape;
		
		
		if(hiddenShape == 0) {
			input2OutputWeights = new double[outputShape][inputShape];
			deltaInput2OutputWeights = new double[outputShape][inputShape];
		} else {

			this.hiddenValues 				= new double[hiddenLayers][hiddenShape][1	];
			this.input2HiddenWeights 		= new double[hiddenShape][inputShape	];
			this.hidden2OutputWeights 		= new double[outputShape][hiddenShape]; 
			this.deltaInput2HiddenWeights 	= new double[hiddenShape][inputShape	];
			this.deltaHidden2OutputWeights 	= new double[outputShape][hiddenShape]; 
			this.hiddenWeights				= new double[hiddenLayers-1][hiddenShape][hiddenShape];
			this.deltaHiddenWeights			= new double[hiddenLayers-1][hiddenShape][hiddenShape];
		}
	}
	
	
	/**
	 * Declares a network with the given shapes. Assumes no hidden Layers
	 * 
	 * @param hiddenShape - number of nodes in layer
	 * @param inputShape - number of nodes in input layer
	 * @param outputShape - number of nodes in output layer
	 */
	public SimpleNeuralNetwork(int inputShape, int outputShape) {
		this.inputValues		= new double[inputShape ][1];
		this.outputValues 	= new double[outputShape][1];
		this.hiddenShape 	= 0;
		
		input2OutputWeights 		= new double[outputShape][inputShape];
		deltaInput2OutputWeights	= new double[outputShape][inputShape];
	}
	
	
	/**
	 * Prepares network for training
	 * 
	 * @param epochs - number of training epochs
	 * @param batchsize - number of samples to process before updating weights
	 * @param learningRate - 
	 */
	public void compile(int epochs, int batchsize, double learningRate) {
		if(epochs <= 0) {
			throw new IllegalArgumentException("epochs must be a positive non-zero value.\n"
					+ "Recieved " + epochs);
		}
		if(batchSize <= 0) {
			throw new IllegalArgumentException("batchSize must be a positive non-zero value.\n"
					+ "Recieved " + batchSize);
		}
		if(learningRate <= 0) {
			throw new IllegalArgumentException("learningRate must be a positive non-zero value.\n"
					+ "Recieved " + learningRate);
		}
		
		this.lossPerEpoch 	= new double[epochs];
		this.epochs 			= epochs;
		this.batchSize 		= batchsize;
		this.learningRate 	= learningRate;
		initializeWeights();
	}
	
	
	/**
	 * Prepares network for training
	 * 
	 * @param epochs - number of training epochs
	 * @param batchsize - number of samples to process before updating weights
	 */
	public void compile(int epochs, int batchsize) {
		if(epochs <= 0) {
			throw new IllegalArgumentException("epochs must be a positive non-zero value.\n"
					+ "Recieved " + epochs);
		}
		if(batchSize <= 0) {
			throw new IllegalArgumentException("batchSize must be a positive non-zero value.\n"
					+ "Recieved " + batchSize);
		}
		
		this.lossPerEpoch = new double[epochs];
		this.epochs 	= epochs;
		this.batchSize 	= batchsize;
		initializeWeights();
	}
	
	
	/**
	 * Prepares network for training
	 * 
	 * @param epochs - number of training epochs
	 */
	public void compile(int epochs) {
		if(epochs <= 0) {
			throw new IllegalArgumentException("epochs must be a positive non-zero value.\n"
					+ "Recieved " + epochs);
		}
		if(batchSize <= 0) {
			throw new IllegalArgumentException("batchSize must be a positive non-zero value.\n"
					+ "Recieved " + batchSize);
		}
		
		this.lossPerEpoch = new double[epochs];
		this.epochs 	= epochs;
		this.batchSize 	= 1;
		initializeWeights();
	}
	
	
	/**
	 * Trains model for the fixed number of epochs defined in the compile method
	 * 
	 * @param samples - training input samples
	 * @param targets - target output samples
	 * @throws Exception
	 */
	public void train(double[][] samples, double targets[][]) throws Exception {
		
		if(samples[0].length != inputValues.length) {
			throw new IllegalArgumentException("sample size must match input shape. \n"
					+ "Recieved samples of length " + samples[0].length + " for input shape " + inputValues.length);
		}
		if(targets[0].length != outputValues.length) {
			throw new IllegalArgumentException("Target size must match output shape. \n"
					+ "Recieved targets of length " + targets[0].length + " for output shape " + outputValues.length);
		}
		
		targetMatrix 	= new double[targets.length][targets[0].length][1];
		predictions 		= new double[targets.length][targets[0].length][1];
		
		for(int i = 0; i < targets.length; i++) {
			targetMatrix[i] = Matrix.fromArray(targets[i], 1);
		}	
		
		for(int i = 0; i < epochs; i++) {
			for(int j = 0; j < samples.length; j++) {
				for(int k = 0; k < batchSize && j+k < samples.length; k++) {
					inputValues = Matrix.fromArray(samples[j+k], 1); 	// Initialize input values
					predictions[j+k] = feedForward();	// Calculate and save output values
					backPropagate(Matrix.fromArray(targets[j+k], 1));	// Calculate weight changes
				}
				
				updateWeights();		// Apply calculated weight changes
			}
			
			lossPerEpoch[i] = loss(predictions, targetMatrix);	// Calculate overall loss of model
			
			System.out.println("Epoch " + (i+1) + " of " + epochs + ": ");
			System.out.println("Loss:\t" + lossPerEpoch[i]);
			System.out.println("=======================================");
		}
		
		trained = true;
	}
	
	
	/**
	 * Calculate node values for current sample
	 * 
	 * @return computed output values 
	 * @throws Exception 
	 */
	public double[][] feedForward() throws Exception {	
		if(hiddenShape > 0) {
			// Calculate values in first hidden layer
			hiddenValues[0] = sigmoid(Matrix.multiply(input2HiddenWeights,	inputValues ));
			
			// Calculate values for remaining hidden layers
			for(int i = 1; i < hiddenValues.length; i++) {
				hiddenValues[i] = sigmoid(Matrix.multiply(hiddenWeights[i-1],	hiddenValues[i-1] ));
			}
			
			// Calculate values for output layer
			outputValues = sigmoid(Matrix.multiply(hidden2OutputWeights,	hiddenValues[hiddenValues.length-1]));
			
		} else {
			outputValues = sigmoid(Matrix.multiply(input2OutputWeights,	inputValues));
		}
		
		return outputValues;
	}


	/**
	 * Computes errors by layer and applies gradient descent to compute weight changes.
	 * 
	 * @param currentTarget - target that matches the sample being currently worked on
	 * @throws Exception 
	 */
	public void backPropagate(double[][] currentTarget) throws Exception {
		double[][] error = Matrix.subtract(currentTarget, outputValues);	// Error of output layer
		double[][] hiddenError;
		
		if(hiddenShape != 0) {
			
			// Gradient Descent
			deltaHidden2OutputWeights =	Matrix.add(deltaHidden2OutputWeights,
					Matrix.multiply(
							Matrix.scalMultiply(
									Matrix.elementMultiply(
											sigmoidSlope(outputValues), error
									),
									learningRate
							),
							Matrix.transpose(hiddenValues[hiddenValues.length-1])
					)
			);

			hiddenError = Matrix.multiply(Matrix.transpose(hidden2OutputWeights), error);	// Error of hidden layer
			
			for(int i = deltaHiddenWeights.length-1; i >= 0; i--) {
				deltaHiddenWeights[i] =	Matrix.add(deltaHiddenWeights[i],
						Matrix.multiply(
								Matrix.scalMultiply(
										Matrix.elementMultiply(
												sigmoidSlope(hiddenValues[i+1]), hiddenError
										),
										learningRate
								),
								Matrix.transpose(hiddenValues[i])
						)
					);
				
				hiddenError = Matrix.multiply(Matrix.transpose(hiddenWeights[hiddenWeights.length-1-i]), hiddenError);
			}
			
			deltaInput2HiddenWeights = 	Matrix.add(deltaInput2HiddenWeights,
					Matrix.multiply(
							Matrix.scalMultiply(
									Matrix.elementMultiply(
											sigmoidSlope(hiddenValues[0]), hiddenError
									), 
									learningRate
							),	
							Matrix.transpose(inputValues)
					)
				);
		} else {
			
			// Gradient Descent
			
			deltaInput2OutputWeights =	Matrix.add(deltaInput2OutputWeights,
					Matrix.multiply(
							Matrix.scalMultiply(
									Matrix.elementMultiply(
											sigmoidSlope(outputValues), error
									),
									learningRate
							),
							Matrix.transpose(inputValues)
					)
				);
		}
	}
	
	
	/**
	 * Computes over all loss of the model
	 * 
	 * @param predictions - array of prediction matrices
	 * @param targetMatrix - array of target matrices
	 * @return
	 */
	private double loss(double[][][] predictions, double[][][] targetMatrix) {
		return MSE(predictions, targetMatrix); 
	}
	
	
	/**
	 * Make prediction on trained model
	 * 
	 * @param sample
	 * @return
	 * @throws Exception
	 */
	public double[][] predict(double[] sample) throws Exception {
		if(!trained) {
			throw new RuntimeException("Model must be compiled and trained before predictions can be made.");
		}
		
		inputValues = Matrix.fromArray(sample, 1);
		return feedForward();
	}
	
	
	/**
	 * Sets all weights to a random real number uniformly in [0, 1)
	 */
	private void initializeWeights() {
		if(hiddenShape != 0) {

			for(int i = 0; i < input2HiddenWeights.length; i++) {
				for(int j = 0; j < input2HiddenWeights[0].length; j++) {
					input2HiddenWeights[i][j] = random.nextDouble();
				}
			}
			
			for(int i = 0; i < hiddenWeights.length; i++) {
				for(int j = 0; j < hiddenWeights[0].length; j++) {
					for(int k = 0; k < hiddenWeights[0][0].length; k++) {
						hiddenWeights[i][j][k] = random.nextDouble();
					}
				}
			}
			
			for(int i = 0; i < hidden2OutputWeights.length; i++) {
				for(int j = 0; j < hidden2OutputWeights[0].length; j++) {
					hidden2OutputWeights[i][j] = random.nextDouble();
				}
			}
		} else {
			for(int i = 0; i < input2OutputWeights.length; i++) {
				for(int j = 0; j< input2OutputWeights[0].length; j++) {
					input2OutputWeights[i][j] = random.nextDouble();
				}
			}
		}
		
	}
	
	
	/**
	 * Reset all weight changes to zero
	 */
	private void resetDeltas() {
		
		if(hiddenShape != 0) {
			for(int i = 0; i < deltaInput2HiddenWeights.length; i++) {
				for(int j = 0; j < deltaInput2HiddenWeights[0].length; j++) {
					deltaInput2HiddenWeights[i][j] = 0;
				}
			}
			
			for(int i = 0; i < deltaHiddenWeights.length; i++) {
				for(int j = 0; j < deltaHiddenWeights[0].length; j++) {
					for(int k = 0; k < deltaHiddenWeights[0][0].length; k++) {
						deltaHiddenWeights[i][j][k] = 0;
					}
				}
			}
			
			for(int i = 0; i < deltaHidden2OutputWeights.length; i++) {
				for(int j = 0; j < deltaHidden2OutputWeights[0].length; j++) {
					deltaHidden2OutputWeights[i][j] = 0;
				}
			}
		} else {
			for(int i = 0; i < deltaInput2OutputWeights.length; i++) {
				for(int j = 0; j < deltaInput2OutputWeights[0].length; j++) {
					deltaInput2OutputWeights[i][j] = 0;
				}
			}
		}
	}
	
	
	/**
	 * Adjust weights of network
	 */
	private void updateWeights() {
		
		if(hiddenShape != 0) {
			input2HiddenWeights = Matrix.add(Matrix.scalDivide(deltaInput2HiddenWeights, batchSize), input2HiddenWeights);
			
			for(int i = 0; i < hiddenWeights.length; i++) {
				hiddenWeights[i] = Matrix.add(Matrix.scalDivide(deltaHiddenWeights[i], batchSize), hiddenWeights[i]);
			}
			
			hidden2OutputWeights = Matrix.add(Matrix.scalDivide(deltaHidden2OutputWeights, batchSize), hidden2OutputWeights);
		} else {
			input2OutputWeights = Matrix.add(Matrix.scalDivide(deltaInput2OutputWeights, batchSize), input2OutputWeights);
		}
		
		
		resetDeltas();
	}
	
	
	/**
	 * Computes sigmoid output for all elements in A
	 * 
	 * TODO: If sigmoid receives Infinity return 1, if negative infinity return 0
	 * 
	 * @param A - mxn matrix
	 * @return an mxn matrix containing sigmoid outputs
	 */
	private double[][] sigmoid(double[][] A) {
		double result[][] = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				if(A[i][j] == Double.POSITIVE_INFINITY) {
					
				} else if(A[i][j] == Double.NEGATIVE_INFINITY) {
					
				} else {	
					result[i][j] = Math.pow(1+Math.exp(-A[i][j]), -1);
				}
			};
		}
		
		return result;
	}
	
	
	/**
	 * Computes sigmoid derivative output for all elements in A
	 * 
	 * If sigmoidSlope receives Infinity or Negative Infinity it will return a zero
	 * 
	 * TODO: Create an independent method for sigmoid and sigmoidSlope that has formula and takes and returns a double
	 * 
	 * @param A - mxn matrix
	 * @return an mxn matrix containing sigmoid derivative outputs
	 */
	private double[][] sigmoidSlope(double[][] A) {
		double result[][] = new double[A.length][A[0].length];
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				if(A[i][j] == Double.POSITIVE_INFINITY || A[i][j] == Double.NEGATIVE_INFINITY) {
					result[i][j] = 0;
				} else {
					result[i][j] = (Math.exp(-A[i][j]))/(Math.pow(1+Math.exp(-A[i][j]), 2));
				}
			}
		}
		
		return result;
	}
	
	
	/**
	 * Computes mean squared error of output vs expected
	 * 
	 * @param output - model output
	 * @param expected - expected output
	 * @return mean squared of provided data sets
	 */
	private double MSE(double[][][] output, double[][][] expected) {
		if(output.length != expected.length || output[0].length != expected[0].length) {
			throw new IllegalArgumentException();
		}
		
		double result = 0.0;
		
		
		
		for(int i = 0; i < output.length; i++) {
			//Matrix.print(output[i]);
			
			for(int j = 0; j < output[0].length; j++) {
				for(int k = 0; k < output[0][0].length; k++) {
					result += Math.pow(expected[i][j][j] - output[i][j][j], 2);
					//System.out.println(expected[i][j][j] +  "\t\t"+  output[i][j][j]);
					//System.out.println("\n\n" + output.length*output[0].length + "\n\n" + result + "\n\n");
				}
			}
		}
		//System.out.println("\n\n" + output.length*output[0].length + "\n\n" + result + "\n\n");
		
		result = result / (output.length*output[0].length);
		
		//System.out.println("\n\n" + output.length*output[0].length + "\n\n" + result + "\n\n");
		
		if(Double.isNaN(result)) {
			System.err.print("Error in MSE");
			System.exit(1);
		}
		
		return result;
	}
	
	
} // end Simple Neural Network



/**
 * The Matrix class contains several methods useful for performing matrix operations.
 */
class Matrix {
	
	/**
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	private static double[] add(double[] a, double[] b, int val) {
		if(a.length != b.length) {
			throw new IllegalArgumentException("length of a and b must match.\n"
					+ "Recieved " + a.length + ", " + b.length);
		}
		
		double[] result = new double[a.length];
		
		for(int i = 0; i < a.length; i++) {
			result[i] = a[i] + val*b[i];
		}
		
		return result;
	}


	public static double[] add(double[] a, double[] b) {
		return add(a, b, 1);
	}
	
	
	/**
	 * computes a-b element-wise
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] subtract(double[] a, double[] b) {
		double[] result = new double[a.length];
		
		for(int i = 0; i < a.length; i++) {
			result[i] = a[i] - b[i];
		}
		
		return result;
		//return add(a, b, -1);
	}
	
	
	/**
	 * 
	 * @param A
	 * @param B
	 * @return
	 */
	public static double[][] add(double[][] A, double[][] B) {
		if(A.length != B.length || A[0].length != B[0].length) {
			throw new IllegalArgumentException("Shape of A and B must match.\n"
					+ "Recieved [" + A.length + ", " + A[0].length + "], [" + B.length + ", " + B[0].length + "]");
		}
		
		double[][] result = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j =0; j < A[0].length; j++) {
				result[i][j] = A[i][j] + B[i][j];
			}
		}
		
		return result;
	}
	
	
	/**
	 * 
	 * @param A
	 * @param B
	 * @return
	 */
	public static double[][] subtract(double[][] A, double[][] B) {
		if(A.length != B.length || A[0].length != B[0].length) {
			throw new IllegalArgumentException("Shape of A and B must match.\n"
					+ "Recieved [" + A.length + ", " + A[0].length + "], [" + B.length + ", " + B[0].length + "]");
		}
		
		double[][] result = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				result[i][j] = A[i][j]-B[i][j];
			}
		}
		
		return result;
	}
	
	
	/**
	 * Calculates multiplication between two matrices
	 * 
	 * @param A - a lxm matrix
	 * @param B - a mxn matrix
	 * @return A lxn matrix that is the matrix product of A and B
	 * @throws Exception
	 */
	public static double[][] multiply(double[][] A, double[][] B) throws Exception {
		if(A[0].length != B.length) {
			throw new Exception("Dimension Mismatch for " + A.length + "x" + A[0].length + " and " + B.length + "x" + B[0].length);
		}
		
		double result[][] = new double[A.length][B[0].length];
		double TransposeB[][] = transpose(B);
		
		for(int i = 0; i < result.length; i++) {
			for(int j = 0; j < result[0].length; j++) {
				result[i][j] = dotProduct(A[i], TransposeB[j]);
			}
		}
		
		return result;
	} // End multiply
	
	
	/**
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] scalMultiply(double[] b, double a) {
		double[] result = new double[b.length];
		
		for(int i = 0; i < b.length; i++) {
			result[i] = a*b[i];
		}
		
		return result;
	}
	
	
	/**
	 * 
	 * @param a
	 * @param B
	 * @return
	 */
	public static double[][] scalMultiply(double[][] B, double a) {
		double[][] result = new double[B.length][B[0].length];
		
		for(int i = 0; i < B.length; i++) {
			result[i] = scalMultiply(B[i], a);
		}
		
		return result; 
	}
	
	
	/**
	 * Computes element wise matrix product (Hadamard product)
	 * 
	 * @param A - mxn matrix
	 * @param B - mxn matrix
	 * @return mxn matrix that is the result of element wise multiplication between A and B
	 */
	public static double[][] elementMultiply(double[][] A, double[][] B) {
		if(A.length != B.length || A[0].length != B[0].length) {
			throw new IllegalArgumentException("Shape of A and B must match.\n"
					+ "Recieved shapes [" + A.length + ", " + A[0].length + "] and [" + B.length + ", " + B[0].length + "]");
		}
		
		double[][] result = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				result[i][j] = A[i][j]*B[i][j];
			}
		}
		
		return result;
	}
	
	
	/**
	 * Computes the transpose of matrix A
	 * 
	 * @param A - a mxn matrix
	 * @return The transpose of matrix A (a nxm matrix)
	 */
	public static double[][] transpose(double[][] A) {
		double[][] result = new double[A[0].length][A.length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				result[j][i] = A[i][j];
			}
		}
		
		return result;
	} // End transpose
	
	
	/**
	 * Computes the dot product of two vectors
	 * 
	 * @param a - vector of dimension n
	 * @param b - vector of dimension n
	 * @return A scalar equal to the dot product of vectors a and b
	 * @throws Exception
	 */
	public static double dotProduct(double[] a, double[] b) throws Exception {
		if(a.length != b.length) {
			throw new Exception("Vectors must be of the same length\n" + 
					"Recieved vectors of size" + a.length + " and " + b.length + ". ");
		}
		
		double result = 0;
		
		for(int i = 0; i < a.length; i++)
			result += a[i]*b[i];
		
		return result;
	} // End dotProduct
	
	
	/**
	 * If axis = 0 then a matrix with 1 column will be returned.
	 * If Axis = 1 then a matrix with 1 row will be returned.
	 * 
	 * @param arr - array of length n
	 * @param axis - axis along which to convert
	 * @return a nx1 or 1xn matrix
	 */
	public static double[][] fromArray(double[] arr, int axis) {
		double[][] result;
		
		if(axis == 0) {
			result = new double[1][arr.length];
			
			for(int i = 0; i < arr.length; i++) {
				result[0][i] = arr[i];
			}
		} else {
			result = new double[arr.length][1];
			
			for(int i = 0; i < arr.length; i++) {
				result[i][0] = arr[i];
			}
		}
		
		return result;
	}
	
	
	/**
	 * Divides a matrix by a scalar value
	 * 
	 * @param A - mxn matrix
	 * @param b - constant value
	 * @return nxm matrix resulting from scaler division
	 */
	public static double[][] scalDivide(double[][] A, double b) {
		double[][] result = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				result[i][j] = A[i][j]/b;
			}
		}
		
		return result;
	}
	
	
	/**
	 * Divides a matrix by a scalar value
	 * 
	 * @param A - lxmxn matrix
	 * @param b - constant value
	 * @return lxnxm matrix resulting from scaler division
	 */
	public static double[][] scalDivide(double[][][] A, double b) {
		double[][] result = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++) {
			for(int j = 0; j < A[0].length; j++) {
				for(int k = 0; k < A[0][0].length; k++) {
					result[i][j] = A[i][j][k]/b;
				}	
			}
		}
		
		return result;
	}
	
	
	/**
	 * Prints a 2d array formatted as if it was a matrix
	 * 
	 * @param arr - array to print
	 */
	public static void print(double[][] arr) {
		for(int i = 0; i < arr.length; i++) {
			System.out.print("[    ");
			for(int j = 0; j < arr[0].length; j++) {
				System.out.format("%.08f     ", arr[i][j]);
			}
			System.out.println("]");
		}
	}
} // End Matrix
