package algorithm;

import java.util.Arrays;
import java.util.Random;

public class BPNeuralNetwork {

	/**
	 * ÿ��Ľ�����������鳤�ȼ�Ϊ������
	 */
	int[] layerNumNodes;

	/**
	 * ���������ڵ�(��Ӧ����ʱֵ)
	 */
	public double[][] layer;

	/**
	 * ��������ڵ����
	 */
	public double[][] layerErr;

	/**
	 * ����ڵ�Ȩ��
	 */
	public double[][][] layer_weight;

	/**
	 * ����ڵ�Ȩ�ض���
	 */
	public double[][][] layer_weight_delta;

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
		rate = paraRate;
		mobp = paraMobp;

		// Step 2. Across layer initialization.
		layer = new double[layerNumNodes.length][];
		layerErr = new double[layerNumNodes.length][];
		// ����Ӧ����layerNumNodes.length - 1
		layer_weight = new double[layerNumNodes.length][][];
		layer_weight_delta = new double[layerNumNodes.length][][];

		// Step 3. Inner layer initialization.
		for (int l = 0; l < layerNumNodes.length; l++) {
			layer[l] = new double[layerNumNodes[l]];
			layerErr[l] = new double[layerNumNodes[l]];

			if (l + 1 < layerNumNodes.length) {
				// ��1��+1Ϊƫ��������
				layer_weight[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				layer_weight_delta[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				for (int j = 0; j < layerNumNodes[l] + 1; j++) {
					for (int i = 0; i < layerNumNodes[l + 1]; i++) {
						// �����ʼ��Ȩ��
						layer_weight[l][j][i] = random.nextDouble();
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
		for (int l = 1; l < layer.length; l++) {
			for (int j = 0; j < layer[l].length; j++) {
				// �����Ȩ��
				double z = layer_weight[l - 1][layer[l - 1].length][j];
				for (int i = 0; i < layer[l - 1].length; i++) {
					// Ҳ���԰�paraIn������layer[0]
					layer[l - 1][i] = l == 1 ? paraIn[i] : layer[l - 1][i];
					z += layer_weight[l - 1][i][j] * layer[l - 1][i];
				}// Of for i

				// Sigmoid�����
				layer[l][j] = 1 / (1 + Math.exp(-z));
			}// Of for j
		}// Of for l
		return layer[layer.length - 1];
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
		int l = layer.length - 1;
		for (int j = 0; j < layerErr[l].length; j++) {
			layerErr[l][j] = layer[l][j] * (1 - layer[l][j])
					* (paraTarget[j] - layer[l][j]);
		}// Of for j

		// Step 2. ��㷴��
		while (l-- > 0) {
			// ����ڵ����
			for (int j = 0; j < layerErr[l].length; j++) {
				double z = 0.0;
				// �����һ���ÿ���ڵ�
				for (int i = 0; i < layerErr[l + 1].length; i++) {
					z = z + l > 0 ? layerErr[l + 1][i] * layer_weight[l][j][i]
							: 0;
					// �����㶯������
					layer_weight_delta[l][j][i] = mobp
							* layer_weight_delta[l][j][i] + rate
							* layerErr[l + 1][i] * layer[l][j];
					// ������Ȩ�ص���
					layer_weight[l][j][i] += layer_weight_delta[l][j][i];
					if (j == layerErr[l].length - 1) {
						// �ؾද������
						layer_weight_delta[l][j + 1][i] = mobp
								* layer_weight_delta[l][j + 1][i] + rate
								* layerErr[l + 1][i];
						// �ؾ�Ȩ�ص���
						layer_weight[l][j + 1][i] += layer_weight_delta[l][j + 1][i];
					}// Of if
				}// Of for i

				// ��¼���
				layerErr[l][j] = z * layer[l][j] * (1 - layer[l][j]);
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
		}//Of for j

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
	}//Of main
}// Of class BPNeuralNetwork
