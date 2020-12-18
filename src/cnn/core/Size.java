package cnn.core;

/**
 * The size of a convolution core.
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
public class Size {
	/**
	 * Cannot be changed after initialization.
	 */
	public final int width;

	/**
	 * Cannot be changed after initialization.
	 */
	public final int height;

	/**
	 *********************** 
	 * The first constructor.
	 * 
	 * @param paraWidth
	 *            The given width.
	 * @param paraHeight
	 *            The given height.
	 *********************** 
	 */
	public Size(int paraWidth, int paraHeight) {
		width = paraWidth;
		height = paraHeight;
	}// Of the first constructor

	/**
	 *********************** 
	 * Divide a scale with another one. For example (4, 12) / (2, 3) = (2, 4).
	 * 
	 * @param paraScaleSize
	 *            The given scale size.
	 * @return The new size.
	 *********************** 
	 */
	public Size divide(Size paraScaleSize) {
		int resultWidth = width / paraScaleSize.width;
		int resultHeight = height / paraScaleSize.height;
		if (resultWidth * paraScaleSize.width != width
				|| resultHeight * paraScaleSize.height != height)
			throw new RuntimeException("Unable to divide " + this + " with "
					+ paraScaleSize);
		return new Size(resultWidth, resultHeight);
	}// Of divide

	/**
	 *********************** 
	 * Subtract a scale with another one, and add a value. For example (4, 12) - (2, 3) + 1 = (3, 10).
	 * 
	 * @param paraScaleSize
	 *            The given scale size.
	 * @param paraAppend
	 *            The appended size to both dimensions.
	 * @return The new size.
	 *********************** 
	 */
	public Size subtract(Size paraScaleSize, int paraAppend) {
		int resultWidth = width - paraScaleSize.width + paraAppend;
		int resultHeight = height - paraScaleSize.height+ paraAppend;
		return new Size(resultWidth, resultHeight);
	}//Of subtract
	
	/**
	 *********************** 
	 * @param The
	 *            string showing itself.
	 *********************** 
	 */
	public String toString() {
		String resultString = "(" + width + ", " + height + ")";
		return resultString;
	}// Of toString

	/**
	 *********************** 
	 * Unit test.
	 *********************** 
	 */
	public static void main(String[] args) {
		Size tempSize1 = new Size(4, 6);
		Size tempSize2 = new Size(2, 2);
		System.out.println("" + tempSize1 + " divide " + tempSize2 + " = "
				+ tempSize1.divide(tempSize2));
		
		System.out.printf("a");

		try {
			new Thread().sleep(1000);
			System.out.println("" + tempSize2 + " divide " + tempSize1 + " = "
					+ tempSize2.divide(tempSize1));
		} catch (Exception ee) {
			System.out.println(ee);
		}//Of try
		
		System.out.println("" + tempSize1 + " - " + tempSize2 + " + 1 = "
				+ tempSize1.subtract(tempSize2, 1));
	}// Of main

}// Of class Size
