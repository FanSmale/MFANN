package algorithm;

import java.util.Arrays;
import java.util.Random;

/**
 * Back-propagation neural networks. The code comes from
 * https://mp.weixin.qq.com
 * /s?__biz=MjM5MjAwODM4MA==&mid=402665740&idx=1&sn=18d84d
 * 72934e59ca8bcd828782172667
 * 
 * @author ��Ԩ
 */

public class BPNeuralNetwork {

	/**
	 * ����
	 */
	int numLayers;

	/**
	 * ÿ��Ľ�����������鳤�ȼ�Ϊ������
	 */
	int[] layerNumNodes;

	/**
	 * ���������ڵ�(��Ӧ����ʱֵ)
	 */
	public double[][] layerNodes;

	/**
	 * ��������ڵ����
	 */
	public double[][] layerNodesErr;

	/**
	 * �������Ȩ��
	 */
	public double[][][] edgeWeights;

	/**
	 * ����ڵ�Ȩ�ض���
	 */
	public double[][][] edgeWeightsDelta;

	/**
	 * ����ϵ��
	 */
	public double mobp;

	/**
	 * ѧϰϵ��
	 */
	public double rate;

	/**
	 * �������������
	 */
	Random random = new Random();

	/**
	 ********************
	 * Constructor.
	 * 
	 * @param paraLayerNumNodes
	 *            The number of nodes for each layer (may be different).
	 * @param paraRate
	 *            Learning rate.
	 * @param paraMobp
	 *            ����ϵ��
	 ********************
	 */
	public BPNeuralNetwork(int[] paraLayerNumNodes, double paraRate,
			double paraMobp) {
		// Step 1. Accept parameters.
		layerNumNodes = paraLayerNumNodes;
		numLayers = layerNumNodes.length;
		rate = paraRate;
		mobp = paraMobp;

		// Step 2. Across layer initialization.
		layerNodes = new double[numLayers][];
		layerNodesErr = new double[numLayers][];
		// ��numLayers��Ϊ��numLayers - 1
		edgeWeights = new double[numLayers - 1][][];
		edgeWeightsDelta = new double[numLayers - 1][][];

		// Step 3. Inner layer initialization.
		for (int l = 0; l < numLayers; l++) {
			layerNodes[l] = new double[layerNumNodes[l]];
			layerNodesErr[l] = new double[layerNumNodes[l]];

			// �ߵĲ����Ƚڵ������1
			if (l + 1 < numLayers) {
				// ��1��+1Ϊƫ��������. layerNumNodes[l]Ϊ�������.
				edgeWeights[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				edgeWeightsDelta[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				for (int j = 0; j < layerNumNodes[l] + 1; j++) {
					for (int i = 0; i < layerNumNodes[l + 1]; i++) {
						// �����ʼ��Ȩ��
						edgeWeights[l][j][i] = random.nextDouble();
					}// Of for i
				}// Of for j
			}// Of if
		}// Of for l
	}// Of the constructor

	/**
	 ********************
	 * �����ǰ����������൱��Ԥ��
	 * 
	 * @param paraIn
	 *            �������ݣ�Ϊһ������������Ϊ���Ը���
	 ********************
	 */
	public double[] computeOut(double[] paraIn) {
		// ��ʼ�������
		for (int i = 0; i < layerNodes[0].length; i++) {
			layerNodes[0][i] = paraIn[i];
		}// Of for i

		// ������ڵ�ֵ
		for (int l = 1; l < numLayers; l++) {
			for (int j = 0; j < layerNodes[l].length; j++) {
				// ��ʼ��Ϊƫ����, ��Ϊ��������Ϊ +1
				double z = edgeWeights[l - 1][layerNodes[l - 1].length][j];
				// �����Ȩ��
				for (int i = 0; i < layerNodes[l - 1].length; i++) {
					// �Ƶ�������ʼλ��, ���ӿɶ���
					// layerNodes[l - 1][i] = l == 1 ? paraIn[i] : layerNodes[l
					// - 1][i];
					// l - 1��ʾ�ߵĲ��, i��ʾ��һ��ڵ��, j��ʾ����ڵ��
					z += edgeWeights[l - 1][i][j] * layerNodes[l - 1][i];
				}// Of for i

				// Sigmoid�����
				layerNodes[l][j] = 1 / (1 + Math.exp(-z));
			}// Of for j
		}// Of for l

		return layerNodes[numLayers - 1];
	}// Of computeOut

	/**
	 ********************
	 * ��㷴��������޸�Ȩ��
	 * 
	 * @param paraTarget
	 *            �������ݣ�Ϊһ������������Ϊ���Ը���
	 ********************
	 */
	public void updateWeight(double[] paraTarget) {
		// Step 1. ��ʼ����������
		int l = numLayers - 1;
		for (int j = 0; j < layerNodesErr[l].length; j++) {
			layerNodesErr[l][j] = layerNodes[l][j] * (1 - layerNodes[l][j])
					* (paraTarget[j] - layerNodes[l][j]);
		}// Of for j

		// Step 2. ��㷴��, l == 0ʱҲ��Ҫ����
		while (l > 0) {
			l--;
			// ��l��, ����ڵ����
			for (int j = 0; j < layerNumNodes[l]; j++) {
				double z = 0.0;
				// �����һ���ÿ���ڵ�
				for (int i = 0; i < layerNumNodes[l + 1]; i++) {
					if (l > 0) {
						z += layerNodesErr[l + 1][i] * edgeWeights[l][j][i];
					}// Of if

					// z = z + l > 0 ? layerNodesErr[l + 1][i] *
					// edgeWeights[l][j][i] : 0;

					// �����㶯������
					edgeWeightsDelta[l][j][i] = mobp
							* edgeWeightsDelta[l][j][i] + rate
							* layerNodesErr[l + 1][i] * layerNodes[l][j];
					// ������Ȩ�ص���
					edgeWeights[l][j][i] += edgeWeightsDelta[l][j][i];
					if (j == layerNumNodes[l] - 1) {
						// �ؾද������
						edgeWeightsDelta[l][j + 1][i] = mobp
								* edgeWeightsDelta[l][j + 1][i] + rate
								* layerNodesErr[l + 1][i];
						// �ؾ�Ȩ�ص���
						edgeWeights[l][j + 1][i] += edgeWeightsDelta[l][j + 1][i];
					}// Of if
					if ((i == 0) && (j == 0)) {
						System.out.println("Layer " + l + " j = " + j
								+ ", edgeWeightsDelta = "
								+ edgeWeightsDelta[l][j + 1][i]);
					}// Of if
				}// Of for i

				// ��¼���
				layerNodesErr[l][j] = layerNodes[l][j] * (1 - layerNodes[l][j])
						* z;
			}// Of for j
		}// Of while
	}// Of updateWeight

	/**
	 ********************
	 * ѵ��.
	 * 
	 * @param paraIn
	 *            �������ݣ�Ϊһ������������Ϊ���Ը���
	 * @param paraTarget
	 *            Ŀ�����ݣ����ǩ
	 ********************
	 */
	public void train(double[] paraIn, double[] paraTarget) {
		double[] out = computeOut(paraIn);
		updateWeight(paraTarget);
	}// Of train

	/**
	 ********************
	 * Test the algorithm.
	 ********************
	 */
	public static void main(String[] args) {
		// ��ʼ��������Ļ�������
		// ��һ��������һ���������飬��ʾ������Ĳ�����ÿ��ڵ�����
		// ����{3,10,10,10,10,2}��ʾ�������3���ڵ㣬�������2���ڵ㣬�м���4�������㣬ÿ��10���ڵ�
		// �ڶ���������ѧϰ�����������������Ƕ���ϵ��
		BPNeuralNetwork bp = new BPNeuralNetwork(new int[] { 2, 10, 2 }, 0.15,
				0.8);

		// �����������ݣ���Ӧ�����4����ά��������
		double[][] data = new double[][] { { 1, 2 }, { 2, 2 }, { 1, 1 },
				{ 2, 1 } };

		// ����Ŀ�����ݣ���Ӧ4���������ݵķ���
		double[][] target = new double[][] { { 1, 0 }, { 0, 1 }, { 0, 1 },
				{ 1, 0 } };

		// ����ѵ��5000��
		for (int n = 0; n < 5000; n++)
			for (int i = 0; i < data.length; i++)
				bp.train(data[i], target[i]);

		// ����ѵ�������������������
		for (int j = 0; j < data.length; j++) {
			double[] result = bp.computeOut(data[j]);
			System.out.println(Arrays.toString(data[j]) + ":"
					+ Arrays.toString(result));
		}// Of for j

		// ����ѵ�������Ԥ��һ�������ݵķ���
		double[] x = new double[] { 3, 1 };

		double[] result = bp.computeOut(x);
		System.out.println("Predict new data");
		System.out.println(Arrays.toString(x) + ":" + Arrays.toString(result));

		// ����ѵ�������Ԥ��һ�������ݵķ���
		x = new double[] { 0.5, 0.5 };

		result = bp.computeOut(x);
		System.out.println("Predict new data");
		System.out.println(Arrays.toString(x) + ":" + Arrays.toString(result));
	}// Of main
}// Of class BPNeuralNetwork
