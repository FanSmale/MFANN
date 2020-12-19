package cnn.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Layer builder.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The convolution neural network project.
 * <p>
 * Progress: Done.<br>
 * Written time: December 18, 2020. <br>
 * Last modify time: December 18, 2020.
 */
public class LayerBuilder {
	/**
	 * Layers.
	 */
	private List<Layer> layers;

	/**
	 *********************** 
	 * The first constructor.
	 *********************** 
	 */
	public LayerBuilder() {
		layers = new ArrayList<Layer>();
	}// Of the first constructor

	/**
	 *********************** 
	 * The second constructor.
	 *********************** 
	 */
	public LayerBuilder(Layer paraLayer) {
		this();
		layers.add(paraLayer);
	}// Of the second constructor

	/**
	 *********************** 
	 * Add a layer.
	 * 
	 * @param paraLayer
	 *            The new layer.
	 *********************** 
	 */
	public void addLayer(Layer paraLayer) {
		layers.add(paraLayer);
	}// Of addLayer
	
	/**
	 *********************** 
	 * Get the specified layer.
	 * 
	 * @param paraIndex
	 *            The index of the layer.
	 *********************** 
	 */
	public Layer getLayer(int paraIndex) throws RuntimeException{
		if (paraIndex >= layers.size()) {
			throw new RuntimeException("Layer " + paraIndex + " is out of range: "
					+ layers.size() + ".");
		}//Of if
		
		return layers.get(paraIndex);
	}//Of getLayer
	
	/**
	 *********************** 
	 * Get the output layer.
	 *********************** 
	 */
	public Layer getOutputLayer() {
		return layers.get(layers.size() - 1);
	}//Of getOutputLayer

	/**
	 *********************** 
	 * Get the number of layers.
	 *********************** 
	 */
	public int getNumLayers() {
		return layers.size();
	}//Of getNumLayers

}// Of class LayerBuilder
