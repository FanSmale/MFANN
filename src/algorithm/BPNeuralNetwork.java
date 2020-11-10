package algorithm;

import java.util.Arrays;
import java.util.Random;

public class BPNeuralNetwork {

	/**
	 * 每层的结点数（本数组长度即为层数）
	 */
	int[] layerNumNodes;

	/**
	 * 神经网络各层节点(对应的临时值)
	 */
	public double[][] layer;

	/**
	 * 神经网络各节点误差
	 */
	public double[][] layerErr;

	/**
	 * 各层节点权重
	 */
	public double[][][] layer_weight;

	/**
	 * 各层节点权重动量
	 */
	public double[][][] layer_weight_delta;

	/**
	 * 动量系数
	 */
	public double mobp;

	/**
	 * 学习系数
	 */
	public double rate;

	/**
	 * 用于生成随机数
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
	 *            动量系数
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
		// 好像应该是layerNumNodes.length - 1
		layer_weight = new double[layerNumNodes.length][][];
		layer_weight_delta = new double[layerNumNodes.length][][];

		// Step 3. Inner layer initialization.
		for (int l = 0; l < layerNumNodes.length; l++) {
			layer[l] = new double[layerNumNodes[l]];
			layerErr[l] = new double[layerNumNodes[l]];

			if (l + 1 < layerNumNodes.length) {
				// 第1个+1为偏移量保留
				layer_weight[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				layer_weight_delta[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				for (int j = 0; j < layerNumNodes[l] + 1; j++) {
					for (int i = 0; i < layerNumNodes[l + 1]; i++) {
						// 随机初始化权重
						layer_weight[l][j][i] = random.nextDouble();
					}// Of for i
				}// Of for j
			}// Of if
		}// Of for l
	}// Of the constructor

	/**
	 ********************
	 * 逐层向前计算输出，相当于预测
	 * 
	 * @param paraIn
	 *            输入数据，为一个向量，长度为属性个数
	 ********************
	 */
	public double[] computeOut(double[] paraIn) {
		for (int l = 1; l < layer.length; l++) {
			for (int j = 0; j < layer[l].length; j++) {
				// 计算加权和
				double z = layer_weight[l - 1][layer[l - 1].length][j];
				for (int i = 0; i < layer[l - 1].length; i++) {
					// 也可以把paraIn拷贝到layer[0]
					layer[l - 1][i] = l == 1 ? paraIn[i] : layer[l - 1][i];
					z += layer_weight[l - 1][i][j] * layer[l - 1][i];
				}// Of for i

				// Sigmoid激活函数
				layer[l][j] = 1 / (1 + Math.exp(-z));
			}// Of for j
		}// Of for l
		return layer[layer.length - 1];
	}// Of computeOut

	/**
	 ********************
	 * 逐层反向计算误差并修改权重
	 * 
	 * @param paraTarget
	 *            输入数据，为一个向量，长度为属性个数
	 ********************
	 */
	public void updateWeight(double[] paraTarget) {
		// Step 1. 初始化输出层误差
		int l = layer.length - 1;
		for (int j = 0; j < layerErr[l].length; j++) {
			layerErr[l][j] = layer[l][j] * (1 - layer[l][j])
					* (paraTarget[j] - layer[l][j]);
		}// Of for j

		// Step 2. 逐层反馈
		while (l-- > 0) {
			// 逐个节点计算
			for (int j = 0; j < layerErr[l].length; j++) {
				double z = 0.0;
				// 针对下一层的每个节点
				for (int i = 0; i < layerErr[l + 1].length; i++) {
					z = z + l > 0 ? layerErr[l + 1][i] * layer_weight[l][j][i]
							: 0;
					// 隐含层动量调整
					layer_weight_delta[l][j][i] = mobp
							* layer_weight_delta[l][j][i] + rate
							* layerErr[l + 1][i] * layer[l][j];
					// 隐含层权重调整
					layer_weight[l][j][i] += layer_weight_delta[l][j][i];
					if (j == layerErr[l].length - 1) {
						// 截距动量调整
						layer_weight_delta[l][j + 1][i] = mobp
								* layer_weight_delta[l][j + 1][i] + rate
								* layerErr[l + 1][i];
						// 截距权重调整
						layer_weight[l][j + 1][i] += layer_weight_delta[l][j + 1][i];
					}// Of if
				}// Of for i

				// 记录误差
				layerErr[l][j] = z * layer[l][j] * (1 - layer[l][j]);
			}// Of for j
		}// Of while
	}// Of updateWeight

	/**
	 ********************
	 * 训练.
	 * 
	 * @param paraIn
	 *            输入数据，为一个向量，长度为属性个数
	 * @param paraTarget
	 *            目标数据，如标签
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
		// 初始化神经网络的基本配置
		// 第一个参数是一个整型数组，表示神经网络的层数和每层节点数，
		// 比如{3,10,10,10,10,2}表示输入层是3个节点，输出层是2个节点，中间有4层隐含层，每层10个节点
		// 第二个参数是学习步长，第三个参数是动量系数
		BPNeuralNetwork bp = new BPNeuralNetwork(new int[] { 2, 10, 2 }, 0.15,
				0.8);

		// 设置样本数据，对应上面的4个二维坐标数据
		double[][] data = new double[][] { { 1, 2 }, { 2, 2 }, { 1, 1 },
				{ 2, 1 } };

		// 设置目标数据，对应4个坐标数据的分类
		double[][] target = new double[][] { { 1, 0 }, { 0, 1 }, { 0, 1 },
				{ 1, 0 } };

		// 迭代训练5000次
		for (int n = 0; n < 5000; n++)
			for (int i = 0; i < data.length; i++)
				bp.train(data[i], target[i]);

		// 根据训练结果来检验样本数据
		for (int j = 0; j < data.length; j++) {
			double[] result = bp.computeOut(data[j]);
			System.out.println(Arrays.toString(data[j]) + ":"
					+ Arrays.toString(result));
		}//Of for j

		// 根据训练结果来预测一条新数据的分类
		double[] x = new double[] { 3, 1 };

		double[] result = bp.computeOut(x);
		System.out.println("Predict new data");
		System.out.println(Arrays.toString(x) + ":" + Arrays.toString(result));

		// 根据训练结果来预测一条新数据的分类
		x = new double[] { 0.5, 0.5 };

		result = bp.computeOut(x);
		System.out.println("Predict new data");
		System.out.println(Arrays.toString(x) + ":" + Arrays.toString(result));
	}//Of main
}// Of class BPNeuralNetwork
