/*
 * SparseClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseClassifier_H_
#define SRC_SparseClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseClassifier {
public:
	SparseClassifier() {
		_dropOut = 0.5;
	}
	~SparseClassifier() {

	}

public:
	int _labelSize;
	int _linearfeatSize;

	dtype _dropOut;
	Metric _eval;

	SparseUniLayer<xpu> _layer_linear;

public:

	inline void init(int labelSize, int linearfeatSize) {
		_labelSize = labelSize;
		_linearfeatSize = linearfeatSize;

		_layer_linear.initial(_labelSize, _linearfeatSize, false, 40, 2);
		_eval.reset();

	}

	inline void release() {
		_layer_linear.release();
	}

	inline dtype process(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;
		int offset = 0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			Tensor<xpu, 2, dtype> output, outputLoss;

			//initialize

			output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
			outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

			//forward propagation
			vector<int> linear_features;

				//random to dropout some feature
			const vector<int>& feature = example.m_features;
			srand(iter * example_num);
			linear_features.clear();
			for (int idy = 0; idy < feature.size(); idy++) {
				if (1.0 * rand() / RAND_MAX >= _dropOut) {
					linear_features.push_back(feature[idy]);
				}
			}

			_layer_linear.ComputeForwardScore(linear_features, output);

			// get delta for each output
			cost += softmax_loss(output, example.m_labels, outputLoss, _eval,
					example_num);

			// loss backward propagation
			_layer_linear.ComputeBackwardLoss(linear_features, output,
					outputLoss);

			//release

			FreeSpace(&output);
			FreeSpace(&outputLoss);

		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	int predict(const vector<int>& features, vector<dtype>& results) {

		Tensor<xpu, 2, dtype> output;

		//initialize

		output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

		//forward propagation

		_layer_linear.ComputeForwardScore(features, output);

		// decode algorithm
		int result = softmax_predict(output, results);

		//release
		FreeSpace(&output);

		return result;

	}

	dtype computeScore(const Example& example) {

		Tensor<xpu, 2, dtype> output;

		//initialize

		output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

		//forward propagation
		_layer_linear.ComputeForwardScore(example.m_features, output);

		// get delta for each output
		dtype cost = softmax_cost(output, example.m_labels);

		//release
		FreeSpace(&output);

		return cost;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_layer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}

	void writeModel();

	void loadModel();

	void checkgrads(const vector<Example>& examples, int iter) {
		checkgrad(this, examples, _layer_linear._W, _layer_linear._gradW,
				"_layer_linear._W", iter, _layer_linear._indexers, false);
		checkgrad(this, examples, _layer_linear._b, _layer_linear._gradb,
				"_layer_linear._b", iter);
	}

public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

};

#endif /* SRC_SparseClassifier_H_ */
