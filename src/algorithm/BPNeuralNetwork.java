package algorithm;

import java.util.Arrays;
import java.util.Random;

/**
 * Back-propagation neural networks. The code comes from
 * https://mp.weixin.qq.com
 * /s?__biz=MjM5MjAwODM4MA==&mid=402665740&idx=1&sn=18d84d
 * 72934e59ca8bcd828782172667
 * 
 * @author 彭渊
 */

public class BPNeuralNetwork {

	/**
	 * 层数
	 */
	int numLayers;

	/**
	 * 每层的结点数（本数组长度即为层数）
	 */
	int[] layerNumNodes;

	/**
	 * 神经网络各层节点(对应的临时值)
	 */
	public double[][] layerNodes;

	/**
	 * 神经网络各节点误差
	 */
	public double[][] layerNodesErr;

	/**
	 * 网络各边权重
	 */
	public double[][][] edgeWeights;

	/**
	 * 各层节点权重动量
	 */
	public double[][][] edgeWeightsDelta;

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
		numLayers = layerNumNodes.length;
		rate = paraRate;
		mobp = paraMobp;

		// Step 2. Across layer initialization.
		layerNodes = new double[numLayers][];
		layerNodesErr = new double[numLayers][];
		// 将numLayers改为了numLayers - 1
		edgeWeights = new double[numLayers - 1][][];
		edgeWeightsDelta = new double[numLayers - 1][][];

		// Step 3. Inner layer initialization.
		for (int l = 0; l < numLayers; l++) {
			layerNodes[l] = new double[layerNumNodes[l]];
			layerNodesErr[l] = new double[layerNumNodes[l]];

			// 边的层数比节点层数少1
			if (l + 1 < numLayers) {
				// 第1个+1为偏移量保留. layerNumNodes[l]为本层点数.
				edgeWeights[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				edgeWeightsDelta[l] = new double[layerNumNodes[l] + 1][layerNumNodes[l + 1]];
				for (int j = 0; j < layerNumNodes[l] + 1; j++) {
					for (int i = 0; i < layerNumNodes[l + 1]; i++) {
						// 随机初始化权重
						edgeWeights[l][j][i] = random.nextDouble();
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
		// 初始化输入层
		for (int i = 0; i < layerNodes[0].length; i++) {
			layerNodes[0][i] = paraIn[i];
		}// Of for i

		// 逐层计算节点值
		for (int l = 1; l < numLayers; l++) {
			for (int j = 0; j < layerNodes[l].length; j++) {
				// 初始化为偏移量, 因为它的输入为 +1
				double z = edgeWeights[l - 1][layerNodes[l - 1].length][j];
				// 计算加权和
				for (int i = 0; i < layerNodes[l - 1].length; i++) {
					// 移到方法开始位置, 增加可读性
					// layerNodes[l - 1][i] = l == 1 ? paraIn[i] : layerNodes[l
					// - 1][i];
					// l - 1表示边的层号, i表示上一层节点号, j表示本层节点号
					z += edgeWeights[l - 1][i][j] * layerNodes[l - 1][i];
				}// Of for i

				// Sigmoid激活函数
				layerNodes[l][j] = 1 / (1 + Math.exp(-z));
			}// Of for j
		}// Of for l

		return layerNodes[numLayers - 1];
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
		int l = numLayers - 1;
		for (int j = 0; j < layerNodesErr[l].length; j++) {
			layerNodesErr[l][j] = layerNodes[l][j] * (1 - layerNodes[l][j])
					* (paraTarget[j] - layerNodes[l][j]);
		}// Of for j

		// Step 2. 逐层反馈, l == 0时也需要计算
		while (l > 0) {
			l--;
			// 第l层, 逐个节点计算
			for (int j = 0; j < layerNumNodes[l]; j++) {
				double z = 0.0;
				// 针对下一层的每个节点
				for (int i = 0; i < layerNumNodes[l + 1]; i++) {
					if (l > 0) {
						z += layerNodesErr[l + 1][i] * edgeWeights[l][j][i];
					}// Of if

					// z = z + l > 0 ? layerNodesErr[l + 1][i] *
					// edgeWeights[l][j][i] : 0;

					// 隐含层动量调整
					edgeWeightsDelta[l][j][i] = mobp
							* edgeWeightsDelta[l][j][i] + rate
							* layerNodesErr[l + 1][i] * layerNodes[l][j];
					// 隐含层权重调整
					edgeWeights[l][j][i] += edgeWeightsDelta[l][j][i];
					if (j == layerNumNodes[l] - 1) {
						// 截距动量调整
						edgeWeightsDelta[l][j + 1][i] = mobp
								* edgeWeightsDelta[l][j + 1][i] + rate
								* layerNodesErr[l + 1][i];
						// 截距权重调整
						edgeWeights[l][j + 1][i] += edgeWeightsDelta[l][j + 1][i];
					}// Of if
					if ((i == 0) && (j == 0)) {
						System.out.println("Layer " + l + " j = " + j
								+ ", edgeWeightsDelta = "
								+ edgeWeightsDelta[l][j + 1][i]);
					}// Of if
				}// Of for i

				// 记录误差
				layerNodesErr[l][j] = layerNodes[l][j] * (1 - layerNodes[l][j])
						* z;
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
		}// Of for j

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
	}// Of main
}// Of class BPNeuralNetwork
