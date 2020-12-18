package cnn.core;

import cnn.utils.MathUtils;

/**
 * One layer, support all four layer types. The code mainly initializes, gets,
 * and sets variables. Essentially no algorithm is implemented.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The convolution neural network project.
 * <p>
 * Progress: The beginning.<br>
 * Written time: December 18, 2020. <br>
 * Last modify time: December 18, 2020.
 */
public class Layer {
	/**
	 * The type of the layer.
	 */
	LayerTypeEnum type;

	/**
	 * The number of out map.
	 */
	int outMapNum;

	/**
	 * The map size.
	 */
	Size mapSize;

	/**
	 * The kernel size.
	 */
	Size kernelSize;

	/**
	 * The scale size.
	 */
	Size scaleSize;

	/**
	 * The index of the class (label) attribute.
	 */
	int classNum = -1;

	/**
	 * Kernel. Dimensions: [front map][out map][width][height].
	 */
	private double[][][][] kernel;

	/**
	 * Bias. The length is outMapNum.
	 */
	private double[] bias;

	/**
	 * Out maps. Dimensions: [batch
	 * size][outMapNum][mapSize.width][mapSize.height].
	 */
	private double[][][][] outMaps;

	/**
	 * Errors. I don't know the dimension yet.
	 */
	private double[][][][] errors;

	/**
	 * For batch processing.
	 */
	private static int recordInBatch = 0;

	/**
	 *********************** 
	 * The first constructor.
	 * 
	 * @param paraNum
	 *            When the type is CONVOLUTION, it is the out map number. when
	 *            the type is OUTPUT, it is the class number.
	 * @param paraSize
	 *            When the type is INPUT, it is the map size; when the type is
	 *            CONVOLUTION, it is the kernel size; when the type is SAMPLING,
	 *            it is the scale size.
	 *********************** 
	 */
	public Layer(LayerTypeEnum paraType, int paraNum, Size paraSize) {
		type = paraType;
		switch (type) {
		case INPUT:
			outMapNum = 1;
			mapSize = paraSize; // No deep copy.
			break;
		case CONVOLUTION:
			outMapNum = paraNum;
			kernelSize = paraSize;
			break;
		case SAMPLING:
			scaleSize = paraSize;
			break;
		case OUTPUT:
			classNum = paraNum;
			mapSize = new Size(1, 1);
			outMapNum = classNum;
			break;
		default:
			System.out
					.println("Internal error occurred in AbstractLayer.java constructor.");
		}// Of switch
	}// Of the first constructor

	/**
	 *********************** 
	 * Initialize the kernel.
	 * 
	 * @param paraNum
	 *            When the type is CONVOLUTION, it is the out map number. when
	 *********************** 
	 */
	public void initKernel(int paraFrontMapNum) {
		kernel = new double[paraFrontMapNum][outMapNum][][];
		for (int i = 0; i < paraFrontMapNum; i++) {
			for (int j = 0; j < outMapNum; j++) {
				kernel[i][j] = MathUtils.randomMatrix(kernelSize.width,
						kernelSize.height, true);
			}// Of for j
		}// Of for i
	}// Of initKernel

	/**
	 *********************** 
	 * Initialize the output kernel. The code is revised to invoke
	 * initKernel(int).
	 *********************** 
	 */
	public void initOutputKernel(int paraFrontMapNum, Size paraSize) {
		kernelSize = paraSize;
		initKernel(paraFrontMapNum);
	}// Of initOutputKernel

	/**
	 *********************** 
	 * Initialize the bias. No parameter. "int frontMapNum" is claimed however
	 * not used.
	 *********************** 
	 */
	public void initBias() {
		bias = MathUtils.randomArray(outMapNum);
	}// Of initBias

	/**
	 *********************** 
	 * Initialize the errors.
	 * 
	 * @param paraBatchSize
	 *            The batch size.
	 *********************** 
	 */
	public void initErrors(int paraBatchSize) {
		errors = new double[paraBatchSize][outMapNum][mapSize.width][mapSize.height];
	}// Of initErrors

	/**
	 *********************** 
	 * Initialize out maps.
	 * 
	 * @param paraBatchSize
	 *            The batch size.
	 *********************** 
	 */
	public void initOutMaps(int paraBatchSize) {
		outMaps = new double[paraBatchSize][outMapNum][mapSize.width][mapSize.height];
	}// Of initOutmaps

	/**
	 *********************** 
	 * Prepare for a new batch.
	 *********************** 
	 */
	public static void prepareForNewBatch() {
		recordInBatch = 0;
	}// Of prepareForNewBatch

	/**
	 *********************** 
	 * Prepare for a new record.
	 *********************** 
	 */
	public static void prepareForNewRecord() {
		recordInBatch++;
	}// Of prepareForNewRecord

	/**
	 *********************** 
	 * Set one value of outMaps.
	 *********************** 
	 */
	public void setMapValue(int paraMapNo, int paraMapWidth, int paraMapHeight,
			double paraValue) {
		outMaps[recordInBatch][paraMapNo][paraMapWidth][paraMapHeight] = paraValue;
	}// Of setMapValue

	/**
	 *********************** 
	 * Set values of the whole map.
	 *********************** 
	 */
	public void setMapValue(int paraMapNo, double[][] paraOutMatrix) {
		outMaps[recordInBatch][paraMapNo] = paraOutMatrix;
	}// Of setMapValue

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public Size getMapSize() {
		return mapSize;
	}// Of getMapSize

	/**
	 *********************** 
	 * Setter.
	 *********************** 
	 */
	public void setMapSize(Size paraMapSize) {
		mapSize = paraMapSize;
	}// Of setMapSize

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public LayerTypeEnum getType() {
		return type;
	}// Of getType

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public int getOutMapNum() {
		return outMapNum;
	}// Of getOutMapNum

	/**
	 *********************** 
	 * Setter.
	 *********************** 
	 */
	public void setOutMapNum(int paraOutMapNum) {
		outMapNum = paraOutMapNum;
	}// Of setOutMapNum

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public Size getKernelSize() {
		return kernelSize;
	}// Of getKernelSize

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public Size getScaleSize() {
		return scaleSize;
	}// Of getScaleSize

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public double[][] getMap(int paraIndex) {
		return outMaps[recordInBatch][paraIndex];
	}// Of getMap

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public double[][] getKernel(int paraFrontMap, int paraOutMap) {
		return kernel[paraFrontMap][paraOutMap];
	}// Of getKernel

	/**
	 *********************** 
	 * Setter. Set one error.
	 *********************** 
	 */
	public void setError(int paraMapNo, int paraMapX, int paraMapY,
			double paraValue) {
		errors[recordInBatch][paraMapNo][paraMapX][paraMapY] = paraValue;
	}// Of setError

	/**
	 *********************** 
	 * Setter. Set one error matrix.
	 *********************** 
	 */
	public void setError(int paraMapNo, double[][] paraMatrix) {
		errors[recordInBatch][paraMapNo] = paraMatrix;
	}// Of setError

	/**
	 *********************** 
	 * Getter. Get one error matrix.
	 *********************** 
	 */
	public double[][] getError(int paraMapNo) {
		return errors[recordInBatch][paraMapNo];
	}// Of getError

	/**
	 *********************** 
	 * Getter. Get the whole error tensor.
	 *********************** 
	 */
	public double[][][][] getErrors() {
		return errors;
	}// Of getErrors

	/**
	 *********************** 
	 * Setter. Set one kernel.
	 *********************** 
	 */
	public void setKernel(int paraLastMapNo, int paraMapNo,
			double[][] paraKernel) {
		kernel[paraLastMapNo][paraMapNo] = paraKernel;
	}// Of setKernel

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public double getBias(int paraMapNo) {
		return bias[paraMapNo];
	}// Of getBias

	/**
	 *********************** 
	 * Setter.
	 *********************** 
	 */
	public void setBias(int paraMapNo, double paraValue) {
		bias[paraMapNo] = paraValue;
	}// Of setBias

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public double[][][][] getMaps() {
		return outMaps;
	}// Of getMaps

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public double[][] getError(int paraRecordId, int paraMapNo) {
		return errors[paraRecordId][paraMapNo];
	}// Of getError

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public double[][] getMap(int paraRecordId, int paraMapNo) {
		return outMaps[paraRecordId][paraMapNo];
	}// Of getMap

	/**
	 *********************** 
	 * Getter.
	 *********************** 
	 */
	public int getClassNum() {
		return classNum;
	}// Of getClassNum

	/**
	 *********************** 
	 * Getter. Get the whole kernel tensor.
	 *********************** 
	 */
	public double[][][][] getKernel() {
		return kernel;
	} // Of getKernel
}// Of class AbstractLayer
