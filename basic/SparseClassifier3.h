/*
 * SparseClassifier3.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseClassifier3_H_
#define SRC_SparseClassifier3_H_

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
class SparseClassifier3 {
public:
  SparseClassifier3() {
    _dropOut = 0.5;
  }
  ~SparseClassifier3() {

  }

public:
  int _labelSize;
  int _linearfeatSize;
  int _hiddenSize1;
  int _hiddenSize2;

  dtype _dropOut;
  Metric _eval;

  SparseUniLayer<xpu> _layer_linear;

  UniLayer<xpu> _hidden_linear;
  UniLayer<xpu> _out_linear;

public:

  inline void init(int labelSize, int linearfeatSize, int hiddenSize1, int hiddenSize2) {
    _labelSize = labelSize;
    _linearfeatSize = linearfeatSize;
    _hiddenSize1 = hiddenSize1;
    _hiddenSize2 = hiddenSize2;

    _layer_linear.initial(_hiddenSize1, _linearfeatSize, false, 40, 2);
    _hidden_linear.initial(_hiddenSize2, _hiddenSize1, false, 50, 2);
    _out_linear.initial(_labelSize, _hiddenSize2, false, 80, 2);
    _eval.reset();

  }

  inline void release() {
    _layer_linear.release();
    _hidden_linear.release();
    _out_linear.release();
  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;

    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      Tensor<xpu, 2, dtype> output, outputLoss;
      Tensor<xpu, 2, dtype> hidden1, hidden1Loss;
      Tensor<xpu, 2, dtype> hidden2, hidden2Loss;

      //initialize

      output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

      hidden1 = NewTensor<xpu>(Shape2(1, _hiddenSize1), d_zero);
      hidden1Loss = NewTensor<xpu>(Shape2(1, _hiddenSize1), d_zero);

      hidden2 = NewTensor<xpu>(Shape2(1, _hiddenSize2), d_zero);
      hidden2Loss = NewTensor<xpu>(Shape2(1, _hiddenSize2), d_zero);

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

      _layer_linear.ComputeForwardScore(linear_features, hidden1);
      _hidden_linear.ComputeForwardScore(hidden1, hidden2);
      _out_linear.ComputeForwardScore(hidden2, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _out_linear.ComputeBackwardLoss(hidden2, output, outputLoss, hidden2Loss);
      _hidden_linear.ComputeBackwardLoss(hidden1, hidden2, hidden2Loss, hidden1Loss);
      _layer_linear.ComputeBackwardLoss(linear_features, hidden1, hidden1Loss);

      //release

      FreeSpace(&output);
      FreeSpace(&outputLoss);
      FreeSpace(&hidden1);
      FreeSpace(&hidden1Loss);
      FreeSpace(&hidden2);
      FreeSpace(&hidden2Loss);

    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const vector<int>& features, vector<dtype>& results) {

    Tensor<xpu, 2, dtype> output;
    Tensor<xpu, 2, dtype> hidden1;
    Tensor<xpu, 2, dtype> hidden2;

    //initialize

    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    hidden1 = NewTensor<xpu>(Shape2(1, _hiddenSize1), d_zero);
    hidden2 = NewTensor<xpu>(Shape2(1, _hiddenSize2), d_zero);

    //forward propagation

    _layer_linear.ComputeForwardScore(features, hidden1);
    _hidden_linear.ComputeForwardScore(hidden1, hidden2);
    _out_linear.ComputeForwardScore(hidden2, output);

    // decode algorithm
    int result = softmax_predict(output, results);

    //release
    FreeSpace(&output);
    FreeSpace(&hidden1);
    FreeSpace(&hidden2);

    return result;

  }

  dtype computeScore(const Example& example) {

    Tensor<xpu, 2, dtype> output;
    Tensor<xpu, 2, dtype> hidden1;
    Tensor<xpu, 2, dtype> hidden2;

    //initialize

    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
    hidden1 = NewTensor<xpu>(Shape2(1, _hiddenSize1), d_zero);
    hidden2 = NewTensor<xpu>(Shape2(1, _hiddenSize2), d_zero);

    //forward propagation
    _layer_linear.ComputeForwardScore(example.m_features, hidden1);
    _hidden_linear.ComputeForwardScore(hidden1, hidden2);
    _out_linear.ComputeForwardScore(hidden2, output);

    // get delta for each output
    dtype cost = softmax_cost(output, example.m_labels);

    //release
    FreeSpace(&output);
    FreeSpace(&hidden1);
    FreeSpace(&hidden2);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _layer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hidden_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _out_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrads(const vector<Example>& examples, int iter) {
    checkgrad(this, examples, _layer_linear._W, _layer_linear._gradW, "_layer_linear._W", iter, _layer_linear._indexers, false);
    checkgrad(this, examples, _layer_linear._b, _layer_linear._gradb, "_layer_linear._b", iter);
    checkgrad(this, examples, _hidden_linear._W, _hidden_linear._gradW, "_hidden_linear._W", iter);
    checkgrad(this, examples, _hidden_linear._b, _hidden_linear._gradb, "_hidden_linear._b", iter);
    checkgrad(this, examples, _out_linear._W, _out_linear._gradW, "_out_linear._W", iter);
    checkgrad(this, examples, _out_linear._b, _out_linear._gradb, "_out_linear._b", iter);
  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

};

#endif /* SRC_SparseClassifier3_H_ */

