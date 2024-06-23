# MIT License
#
# Copyright (c) 2021 Hryhorii Chereda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tensorflow.python.ops import gen_nn_ops
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

from tensorflow.python.ops.linalg.sparse import sparse as tfsp
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, MaxPooling2D, Flatten, AveragePooling1D, MaxPooling1D
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import scipy
from lib import graph
import time


def get_sparse_tf(L):
    """Transforms L matrix in scipy csr to SpareTensor of Tensorflow."""
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.sparse.SparseTensor(indices, L.data, L.shape)
    return tf.sparse.reorder(L)


class GraphLayerwiseRelevancePropagation:
    """
    Class encapsulates functionality for running layer-wise relevance propagation on Graph CNN.
    """

    def __init__(self, model, L=None, K=None, p=None):
        """
        Initialization of internals for relevance computations.
        :param model: gcnn model to LRP procedure is applied on
        :param samples: samples to calculate relevance on, num of samples <= models batch size
        :param labels: used as values for the gcnn output nodes to propagate relevances, num of labels == models batch size
        """
        self.epsilon = 1e-10  # for numerical stability

        # weights = model.get_weights()
        #print(weights)
        # self.activations = model.activations

        self.model = model

        self.activations = [layer.output for layer in model.layers]

        # for layer in model.layers:
        #     w_b = layer.get_weights()
        #     # print(layer.name)
        #     # print("\tw_b:", len(w_b))
        #     act = layer.output
        #     # print("\tact:", act.shape, act.name)
            # print("\tweights\n:", weights.shape)
            # print("\tbiases\n:", bias.shape)

        #self.model.graph._unsafe_unfinalize()  # the computational graph of the model will be modified
        # self.samples = samples
        # self.X = self.activations[0]  # getting the first
        #print("inside GLRP constructor, self.X.shape: ", self.X.shape)


        self.act_weights = {}  # example in this dictionary "conv1": [weights, bias]

        for layer in model.layers:
            name = layer.name
            w_and_b = layer.get_weights()
            if len(w_and_b):
                self.act_weights[name] = w_and_b
                # print(name, w_and_b[0].shape)


        # for act in self.activations[1:]:
        #     w_and_b = []  # 2 element list of weight and bias of one layer.
        #     name = act.name.split('/')
        #     # print(name)
        #     for wt in weights:
        #         #print(wt.name)
        #         if name[0] == wt.name.split('/')[0]:
        #             w_and_b.append(wt)
        #     if w_and_b and (name[0] not in self.act_weights):
        #         self.act_weights[name[0]] = w_and_b

        self.activations.reverse()
        self.X = model.layers[0].input
        self.activations.append(model.layers[0].input)
        # print("\n", "activations:", "\n")
        # for i in range(0, len(self.activations)):
        #     print(self.activations[i].name)


        # !!!
        # first convolutional layer filters
        # self.filters_gc1 = []
        # self.filtered_signal_gc1 = []

        # for layer in model.layers:
        #     name = layer.name
        #     if cheb_conv
        #     print("\t layer name", name)

        start = time.time()
        self.polynomials = []
        self.p = []
        print("\n\tPolynomials of Laplace Matrices")
        print(L)
        print(K)
        print(p)
        if L and K and p:
            L_list = []
            j = 0
            for pp in p:
                L_list.append(L[j])
                j += int(np.log2(pp)) if pp > 1 else 0
            print("\n\t L and K are present, Calculating Polynomials of Laplace Matrices...", end=" ")
            self.polynomials = [self.calc_Laplace_Polynom_old(lap, K=K[i]) for i, lap in enumerate(L_list)]
            self.p = p
        end = time.time()
        print("Time: ", end - start, "\n")


    def get_relevances(self, samples, labels):
        """
        Computes relevances based on input samples.
        :param rule: the propagation rule of the first layer
        :return: the list of relevances, corresponding to different layers of the gcnn.
        The last element of this list contains input relevances and has shape (batch_size, num_of_input_features)
        """

        # Backpropagate softmax value
        # relevances = [tf.nn.softmax(self.activations[0])*tf.cast(self.y, tf.float32)]

        # Backpropagate a value from given labels y
        relevances = [labels]
        #relevances = []

        if len(self.p)!=0 and len(self.polynomials)!=0:
            loc_poly = [pol for pol in self.polynomials]
            loc_pooling = [p for p in self.p]

        print("\n\n\tRelevance calculation:")
        for i in range(0, len(self.activations)-1):  # Need this to provide correct activation[i] with weights [i+1].
            name = self.activations[i].name.split('/')[0].split("_")
            # print("\tname:", name)
            keras_function = K.function(self.model.input, self.activations[i+1])
            activation_values = keras_function(samples)
            # print("\n\tactivation_values", activation_values)
            # print("\n\tactivation_values", type(activation_values))

            #outputs.append(keras_function([self.samples, 1]))
            if 'dense' in name[0]: #or 'fc' in name[0]:
                print("\tFully connected:", name[0])
                print(self.activations[i+1])
                relevances.append(self.prop_fc("_".join(name), activation_values, relevances[-1]))

            elif 'dropout' in name[0]: #or 'fc' in name[0]:
                print("\tDropout layer:", name, ": omitting")
                continue
                # relevances.append(self.prop_fc("_".join(name), activation_values, relevances[-1]))

            elif 'flatten' in name[0]:
                print("\tFlatten layer:", name[0])
                relevances.append(self.prop_flatten(activation_values, relevances[-1]))
                # print("\n")
            elif 'pooling1d' in name[1]:
                # TODO: incorporate pooling type and value into name
                print("\tPooling:", "_".join(name))
                p = loc_pooling.pop()
                if "average" in name[0]:
                    relevances.append(self.prop_avg_pool(activation_values, relevances[-1], ksize=[1, p, 1, 1],
                                                      strides=[1, p, 1, 1]))
                elif "max" in name[0]:
                    relevances.append(self.prop_max_pool(activation_values, relevances[-1], ksize=[1, p, 1, 1],
                                                         strides=[1, p, 1, 1]))
                else:
                    raise Exception('Error parsing the pooling type')

                # print(relevances[-1])
                # import sys
                # sys.exit()

            elif 'conv' in name[1]:
                if len(loc_poly) > 1:
                    print("\tConvolution: ", "_".join(name)) #, "\n")
                    pol = loc_poly.pop()
                    relevances.append(self.prop_gconv("_".join(name), activation_values, relevances[-1], polynomials=pol))
                else:
                    print("\tConvolution, the first layer:", "_".join(name)) #, "\n")
                    pol = loc_poly.pop()
                    #print("\n\tactivation as input, name and shape", self.activations[i + 1], activation_values.shape)
                    relevances.append(self.prop_gconv_first_conv_layer_general("_".join(name), activation_values, relevances[-1],
                                                                      polynomials=pol))
            else:
                raise 'Error parsing layer'

        return relevances[-1].numpy()

    def prop_fc(self, name, activation, relevance):
        """Propagates relevances through fully connected layers."""
        # print("\n\tactivation_fc", type(activation))

        w = self.act_weights[name][0]
        # print("\n\tweight_fc", w.shape)
        # print("weight", w)

        # b = self.act_weights[name][1]  # bias
        w_pos = tf.maximum(0.0, w)
        # if name == "fc1":
        #     print("\t\t"+name+" " + "input == tf.ones_like")
        #     activation = tf.ones_like(activation)
        z = tf.matmul(activation, w_pos) + self.epsilon
        s = relevance / z
        c = tf.matmul(s, tf.transpose(w_pos))
        return c * activation

    def prop_flatten(self, activation, relevance):
        """Propagates relevances from the fully connected part to convolutional part."""
        # print("activation", activation)
        # shape = activation.get_shape().as_list()
        # print("activation", activation.shape)
        # print("\n\tactivation_fc", type(activation))

        return tf.reshape(relevance, tf.shape(activation))

    def prop_avg_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through avg pooling."""
        #print("\n\tactivation_average_pool", type(activation))

        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        # print("activation", activation.shape, type(activation))
        # print("act", act.shape, type(act))
        z = tf.nn.avg_pool(act, ksize, strides, padding='SAME') + self.epsilon
        # print("z", z.shape)
        rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.avg_pool_grad(tf.shape(act), s, ksize, strides, padding='SAME')
        tmp = c * act
        # print("pooling tmp", tmp.shape)
        return tf.squeeze(tmp, [3])

    def prop_max_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through max pooling."""
        #print("\n\tactivation_max_pool", type(activation))
        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        z = tf.nn.max_pool(act, ksize, strides, padding='SAME') + self.epsilon
        # print("...z", np.abs(z.numpy()).min())
        rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.max_pool_grad_v2(act, z, s, ksize, strides, padding='SAME')
        tmp = c * act
        return tf.squeeze(tmp, [3])

    def prop_gconv(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through Graph Convolutional Layers.
        All essential operations are in SCIPY.
        """
        start = time.time()
        W = self.act_weights[name][0]  # weight
        # print("prop_gconv", type(W))

        b = self.act_weights[name][1]  # bias
        # print("\nInside gconv")
        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape, type(W[0,0]))
        # activation
        N, M, Fin = activation.shape
        # N = tf.shape(activation)[0]
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = W.shape[-1]
        #print("\n\tname, inside of the prop_gconv:", name)
        # print("N, M, Fin, Fout", N, M, Fin, Fout)
        K = int(W.shape[0] / Fin)

        W = np.reshape(W, (int(W.shape[0] / K), K, Fout))
        W = np.transpose(W, (1, 0, 2))  # K x Fin x Fout

        # print("relevance_before_calculation_sum\n", relevance.numpy().sum(axis=(1, 2)))
        # print("")
        # print("W shape", W.shape)
        # print("Activation after running, shape", activation.shape)
        # print("Relevance shape", relevance.shape)
        #
        # ## !!! no need in that anymore
        # ## polynomials = self.polynomials
        # print("polynom order_K:", K)
        # print("Polynomials shape", type(polynomials))
        #
        # print("W shape before loop", W.shape)

        rel = tf.zeros(shape=[N, M * Fin]) #, dtype=np.float32)
        # polynomials[0] = polynomials[0]*W[:, :, i]
        # for i in range(1, K):

        for i in range(0, Fout):
            w_pos = polynomials.dot(W[:, :, i])
            # print("w_pos", w_pos.shape, type(w_pos[0,0]))
            # print("i in the loop, scipy", i)
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, W[:, :, i])
            w_pos = tf.maximum(0.0, w_pos)
            w_pos = tf.reshape(w_pos, [M, M, Fin])
            w_pos = tf.transpose(w_pos, perm=[0, 2, 1]) # perm=[0, 2, 1]) axes=[0, 2, 1]  # M x Fin x M
            w_pos = tf.reshape(w_pos, [M * Fin, M])
            # print("i in the loop, tf.reshape", i)
            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            # print("i in the loop, tf.reshape", i)
            z = tf.matmul(activation, w_pos) + self.epsilon  # N x M
            # print("i in the loop, tf.matmul(activation, w_pos)", i)
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            # print("i in the loop, tf.matmul(s, np.transpose(w_pos))", i)
            rel += c * activation
        end = time.time()
        # #
        rel = tf.reshape(rel, [N, M, Fin])
        print("\n\t" + name + ",", "relevance propagation time is: ", end - start)

        return rel


    def prop_gconv_first_conv_layer_general(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        #print(b)
        print("\tcomparison of biases", b.shape, b[0,0,16])

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        # activation = np.ones_like(activation)

        eps = 1e-10
        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w = polynomials.dot(W[:, i])
            w = tf.reshape(w, [M, M])
            # w_pos = polynomials.dot(W[:, i])
            # w_pos = tf.maximum(0.0, w_pos)
            # w_pos = tf.reshape(w_pos, [M, M])

            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            # !!!
            # general rule for relevance redistribution

            # !!!
            # TODO: generalize for biases per one vertex
            # currently b[0, 0, i] one scalar bias per filter
            z = tf.matmul(activation, w) + b[0, 0, i]# self.epsilon     # N x M
            z = z - eps + tf.cast(tf.math.greater_equal(z, tf.zeros(shape=z.shape, dtype=np.float32)), dtype=np.float32) * 2 * eps #self.epsilon

            # z = z - non_zero * self.eps
            print("z", np.abs(z.numpy()).min()) #, z.numpy().max())
            s = relevance[:, :, i] / z  # N x M
            print("s", s.numpy().min(), s.numpy().max())
            c = tf.matmul(s, tf.transpose(w))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel

    def prop_gconv_first_conv_layer_constraints(self, name, activation, relevance, polynomials, lb=0, hb=20):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        lb = tf.zeros(shape=[N, M], dtype=np.float32) + lb
        hb = tf.zeros(shape=[N, M], dtype=np.float32) + hb

        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w = polynomials.dot(W[:, i])
            w = tf.reshape(w, [M, M])
            w_pos = tf.maximum(0.0, w)
            w_neg = tf.minimum(0.0, w)
            w_pos = tf.reshape(w_pos, [M, M])
            w_neg = tf.reshape(w_neg, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M

            # !!!
            # Z^B rule

            z = tf.matmul(activation, w) - tf.matmul(lb, w_pos) - tf.matmul(hb, w_neg) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w))  # N x M by transpose(M * Fin, M) = N x M * Fin
            cp = tf.matmul(s, tf.transpose(w_pos))
            cn = tf.matmul(s, tf.transpose(w_neg))
            rel += c * activation - lb * cp - hb * cn
        end = time.time()

        print("\n\t" + name + ",", "Constraints, relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        # print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def prop_gconv_first_conv_layer(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w_pos = polynomials.dot(W[:, i])
            w_pos = tf.maximum(0.0, w_pos)
            w_pos = tf.reshape(w_pos, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            # !!!
            # Z^+ rule
            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = tf.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        # print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def prop_gconv_first_conv_layer_w_w_rule(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            w = polynomials.dot(W[:, i])
            # w_pos = tf.maximum(0.0, w_pos)
            w = tf.reshape(w, [M, M])

            # !!!
            # w^2 rule
            w_square = w * w
            w_denominator = tf.math.reduce_sum(w_square, axis=0)
            print("w_denominator", w_denominator.shape)
            rel += np.matmul(relevance[:, :, i], w_square) / w_denominator

        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def prop_gconv_first_conv_layer_old(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))


        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w_pos = polynomials.dot(W[:, i])
            w_pos = tf.maximum(0.0, w_pos)
            w_pos = tf.reshape(w_pos, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            # !!!
            # Z^+ rule
            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = tf.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        # print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel

    def prop_gconv_first_conv_layer_flat(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        activation = tf.squeeze(tf.ones_like(activation))

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))


        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            # w_pos = polynomials.dot(W[:, i])
            # w_pos = tf.maximum(0.0, w_pos)
            # w_pos = tf.reshape(w_pos, [M, M])

            w_pos = tf.ones(shape=[M, M])
            # !!!
            # Z^+ rule

            z = tf.matmul(activation, w_pos)  # + self.epsilon  # N x M

            # z = tf.ones_like(activation)
            print("z shape", z.shape)
            s = relevance[:, :, i] / z  # N x M
            r = relevance[:, :, i]
            print("r", r.numpy().min(), r.numpy().max())
            c = tf.matmul(s, tf.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation

        # rel = tf.math.reduce_sum(relevance, axis=2)
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def calc_Laplace_Polynom(self, L, K):
        """
        Calculate Chebyshev polynoms based on Laplace matrix.
        :param L: Laplace matrix M*M
        :param K: Number of polynoms with degree from 0 to K-1
        :return: Chebyshev Polynoms in scipy.sparse.coo, shape (M*M, K)
        """
        # N, M, Fin = self.X.get_shape()
        M = int(L.shape[0])
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.

        # print("\n\ttype L", type(L))


        def get_sparse_tf(L):
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            L = tf.sparse.SparseTensor(indices, L.data, L.shape)
            return tf.sparse.reorder(L)


        def get_CSR_tf(L):
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            L = tf.sparse.SparseTensor(indices, L.data, L.shape)
            L = tf.sparse.reorder(L)
            return tfsp.CSRSparseMatrix(L)


        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)

        #input_L = tf.keras.Input(name='laplace_sparse', sparse=True, shape=(M, M), dtype=tf.dtypes.float32)
        # print("input_L", input_L)
        # print(tf.shape(input_L))
        # input_L = tf.sparse.reshape(input_L, (tf.shape(input_L)[1], tf.shape(input_L)[2]))
        L0 = get_sparse_tf(L)
        L1 = get_CSR_tf(L)
        # L0 = tf.sparse.reorder(input_L)
        #
        # L1 = tfsp.CSRSparseMatrix(tf.sparse.reorder(input_L))

        def concat(x, x_):
            # x_ = tf.sparse.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.sparse.concat(sp_inputs=[x, x_], axis=1)  # K x M x Fin*N

        polynomials = []
        # instead of 2
        Zwei = get_CSR_tf(2 * scipy.sparse.identity(M, dtype=np.float32, format="coo"))
        Minus = get_CSR_tf((-1) * scipy.sparse.identity(M, dtype=np.float32, format="coo"))
        if K > 1:
            # only rank 2 for sparse_tensor_dense_matmul
            T0 = get_CSR_tf(scipy.sparse.identity(M, dtype=np.float32, format="coo"))
            I = get_sparse_tf(scipy.sparse.identity(M, dtype=np.float32, format="coo"))
            T1 = L1
            # polynomials.extend([I, T1]) # the first matrix is I matrix
            tmp = [tf.sparse.reshape(I, shape=(M * M, 1)), tf.sparse.reshape(L0, shape=(M * M, 1))]
            # polynomials = concat(tf.sparse.reshape(I, shape=(M * M, 1)), tf.sparse.reshape(L0, shape=(M * M, 1)))
            # polynomials = tf.sparse.concat(axis=1, sp_inputs = tmp)
            polynomials = tmp
            # print("\n", polynomials.shape)
            # print(polynomials)
            # polynomials = scipy.sparse.hstack([I.reshape(M * M, 1), T1.reshape(M * M, 1)])
        for k in range(2, K):
            # T2 = 2 * tfsp.matmul(L1, T1) - T0  #2 * tfsp.matmul(Zwei, tfsp.matmul(L1, T1)) - T0
            tmp0 = tfsp.matmul(L1, T1)
            # print(Minus.shape)
            # print(T1.shape)
            # print(L1.shape)
            # print(tmp0.shape)
            tmp1 = tfsp.matmul(Zwei, tmp0).to_sparse_tensor()
            tmp2 = tfsp.matmul(Minus, T0).to_sparse_tensor()
            T2 = tf.sparse.add(tmp1, tmp2)  #
            # polynomials = concat(polynomials, tf.sparse.reshape(T2, shape=(M * M, 1)))
            polynomials.append(tf.sparse.reshape(T2, shape=(M * M, 1)))
            # print("Poly K", polynomials.shape)
            T0, T1 = T1, tfsp.CSRSparseMatrix(T2)

        print(type(polynomials[0]))
        print(polynomials[0].shape)
        polynomials = tf.sparse.concat(axis=1, sp_inputs = polynomials)
        # print(polynomials[0])
        # spec_model = tf.keras.Model(inputs=input_L, outputs=polynomials)
        return polynomials



    # def calc_Laplace_Polynom_old(self, L, K):
    def calc_Laplace_Polynom_old(self, L, K):
        """
        Calculate Chebyshev polynoms based on Laplace matrix.
        :param L: Laplace matrix M*M
        :param K: Number of polynoms with degree from 0 to K-1
        :return: Chebyshev Polynoms in scipy.sparse.coo, shape (M*M, K)
        """
        # N, M, Fin = self.X.get_shape()
        M = int(L.shape[0])
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.

        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        polynomials = []
        if K > 1:
            # only rank 2 for sparse_tensor_dense_matmul
            T0 = scipy.sparse.identity(M, dtype=np.float32, format="csr")
            I = scipy.sparse.identity(M, dtype=np.float32, format="csr")
            T1 = L
            # polynomials.extend([I, T1]) # the first matrix is I matrix
            # polynomials = scipy.sparse.hstack([I.reshape(M * M, 1), T1.reshape(M * M, 1)])
            polynomials = [I.reshape(M * M, 1), T1.reshape(M * M, 1)]
        for k in range(2, K):
            T2 = 2 * L * T1 - T0  #
            # polynomials = scipy.sparse.hstack([polynomials, T2.reshape(M * M, 1)])
            polynomials.append(T2.reshape(M * M, 1))
            T0, T1 = T1, T2
        # print(polynomials[0].shape)
        # print(len(polynomials))
        polynomials = scipy.sparse.hstack(polynomials)
        polynomials = polynomials.tocsr()
        # print(type(polynomials))
        # polynomials = polynomials.tocoo()
        # indices = np.column_stack((polynomials.row, polynomials.col))
        # polynomials = tf.sparse.SparseTensor(indices, polynomials.data, polynomials.shape)
        # return tf.sparse.reorder(polynomials)
        return polynomials


class GraphLayerwiseRelevancePropagationLRPeps:
    """
    Class encapsulates functionality for running layer-wise relevance propagation on Graph CNN.
    """

    def __init__(self, model, L=None, K=None, p=None):
        """
        Initialization of internals for relevance computations.
        :param model: gcnn model to LRP procedure is applied on
        :param samples: samples to calculate relevance on, num of samples <= models batch size
        :param labels: used as values for the gcnn output nodes to propagate relevances, num of labels == models batch size
        """
        self.epsilon = 1e-5  # for numerical stability

        # weights = model.get_weights()
        #print(weights)
        # self.activations = model.activations

        self.model = model

        self.activations = [layer.output for layer in model.layers]

        # for layer in model.layers:
        #     w_b = layer.get_weights()
        #     # print(layer.name)
        #     # print("\tw_b:", len(w_b))
        #     act = layer.output
        #     # print("\tact:", act.shape, act.name)
            # print("\tweights\n:", weights.shape)
            # print("\tbiases\n:", bias.shape)

        #self.model.graph._unsafe_unfinalize()  # the computational graph of the model will be modified
        # self.samples = samples
        # self.X = self.activations[0]  # getting the first
        #print("inside GLRP constructor, self.X.shape: ", self.X.shape)


        self.act_weights = {}  # example in this dictionary "conv1": [weights, bias]

        for layer in model.layers:
            name = layer.name
            w_and_b = layer.get_weights()
            if len(w_and_b):
                self.act_weights[name] = w_and_b
                # print(name, w_and_b[0].shape)


        # for act in self.activations[1:]:
        #     w_and_b = []  # 2 element list of weight and bias of one layer.
        #     name = act.name.split('/')
        #     # print(name)
        #     for wt in weights:
        #         #print(wt.name)
        #         if name[0] == wt.name.split('/')[0]:
        #             w_and_b.append(wt)
        #     if w_and_b and (name[0] not in self.act_weights):
        #         self.act_weights[name[0]] = w_and_b

        self.activations.reverse()
        self.X = model.layers[0].input
        self.activations.append(model.layers[0].input)
        # print("\n", "activations:", "\n")
        # for i in range(0, len(self.activations)):
        #     print(self.activations[i].name)


        # !!!
        # first convolutional layer filters
        # self.filters_gc1 = []
        # self.filtered_signal_gc1 = []

        # for layer in model.layers:
        #     name = layer.name
        #     if cheb_conv
        #     print("\t layer name", name)

        start = time.time()
        self.polynomials = []
        print("\n\tPolynomials of Laplace Matrices")
        if L and K and p:
            L_list = []
            j = 0
            for pp in p:
                L_list.append(L[j])
                j += int(np.log2(pp)) if pp > 1 else 0
            print("\n\t L and K are present, Calculating Polynomials of Laplace Matrices...", end=" ")
            self.polynomials = [self.calc_Laplace_Polynom_old(lap, K=K[i]) for i, lap in enumerate(L_list)]
            self.p = p
        end = time.time()
        print("Time: ", end - start, "\n")


    def get_relevances(self, samples, labels):
        """
        Computes relevances based on input samples.
        :param rule: the propagation rule of the first layer
        :return: the list of relevances, corresponding to different layers of the gcnn.
        The last element of this list contains input relevances and has shape (batch_size, num_of_input_features)
        """

        # Backpropagate softmax value
        # relevances = [tf.nn.softmax(self.activations[0])*tf.cast(self.y, tf.float32)]

        # Backpropagate a value from given labels y
        relevances = [labels]
        #relevances = []

        loc_poly = [pol for pol in self.polynomials]
        loc_pooling = [p for p in self.p]
        print("\n\n\tRelevance calculation:")
        for i in range(0, len(self.activations)-1):  # Need this to provide correct activation[i] with weights [i+1].
            name = self.activations[i].name.split('/')[0].split("_")
            # print("\tname:", name)
            keras_function = K.function(self.model.input, self.activations[i+1])
            activation_values = keras_function(samples)
            # print("\n\tactivation_values", activation_values)
            # print("\n\tactivation_values", type(activation_values))

            #outputs.append(keras_function([self.samples, 1]))
            if 'dense' in name[0]: #or 'fc' in name[0]:
                print("\tFully connected:", name[0])
                relevances.append(self.prop_fc("_".join(name), activation_values, relevances[-1]))

            elif 'dropout' in name[0]: #or 'fc' in name[0]:
                print("\tDropout layer:", name, ": omitting")
                continue
                # relevances.append(self.prop_fc("_".join(name), activation_values, relevances[-1]))

            elif 'flatten' in name[0]:
                print("\tFlatten layer:", name[0])
                relevances.append(self.prop_flatten(activation_values, relevances[-1]))
                # print("\n")
            elif 'pooling1d' in name[1]:
                # TODO: incorporate pooling type and value into name
                print("\tPooling:", "_".join(name))
                p = loc_pooling.pop()
                if "average" in name[0]:
                    relevances.append(self.prop_avg_pool(activation_values, relevances[-1], ksize=[1, p, 1, 1],
                                                      strides=[1, p, 1, 1]))
                elif "max" in name[0]:
                    relevances.append(self.prop_max_pool(activation_values, relevances[-1], ksize=[1, p, 1, 1],
                                                         strides=[1, p, 1, 1]))
                else:
                    raise Exception('Error parsing the pooling type')

                # print(relevances[-1])
                # import sys
                # sys.exit()

            elif 'conv' in name[1]:
                if len(loc_poly) > 1:
                    print("\tConvolution: ", "_".join(name)) #, "\n")
                    pol = loc_poly.pop()
                    relevances.append(self.prop_gconv("_".join(name), activation_values, relevances[-1], polynomials=pol))
                else:
                    print("\tConvolution, the first layer:", "_".join(name)) #, "\n")
                    pol = loc_poly.pop()
                    #print("\n\tactivation as input, name and shape", self.activations[i + 1], activation_values.shape)
                    relevances.append(self.prop_gconv_first_conv_layer_general("_".join(name), activation_values, relevances[-1],
                                                                      polynomials=pol))
            else:
                raise 'Error parsing layer'

        return relevances[-1].numpy()

    def prop_fc(self, name, activation, relevance):
        """Propagates relevances through fully connected layers."""
        # print("\n\tactivation_fc", type(activation))

        w = self.act_weights[name][0]
        # print("\n\tweight_fc", w.shape)
        # print("weight", w)

        # b = self.act_weights[name][1]  # bias
        # w_pos = tf.maximum(0.0, w)
        # if name == "fc1":
        #     print("\t\t"+name+" " + "input == tf.ones_like")
        #     activation = tf.ones_like(activation)
        z = tf.matmul(activation, w) + self.epsilon
        z = z - tf.cast(tf.less_equal(z, tf.zeros(shape=z.shape, dtype=np.float32)), dtype=np.float32) * 2 * self.epsilon
        s = relevance / z
        c = tf.matmul(s, tf.transpose(w))
        return c * activation

    def prop_flatten(self, activation, relevance):
        """Propagates relevances from the fully connected part to convolutional part."""
        # print("activation", activation)
        # shape = activation.get_shape().as_list()
        # print("activation", activation.shape)
        # print("\n\tactivation_fc", type(activation))

        return tf.reshape(relevance, tf.shape(activation))

    def prop_avg_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through avg pooling."""
        #print("\n\tactivation_average_pool", type(activation))

        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        # print("activation", activation.shape, type(activation))
        # print("act", act.shape, type(act))
        z = tf.nn.avg_pool(act, ksize, strides, padding='SAME')#  + self.epsilon
        # print("z", z.shape)
        rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.avg_pool_grad(tf.shape(act), s, ksize, strides, padding='SAME')
        tmp = c * act
        # print("pooling tmp", tmp.shape)
        return tf.squeeze(tmp, [3])

    def prop_max_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through max pooling."""
        #print("\n\tactivation_max_pool", type(activation))
        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        z = tf.nn.max_pool(act, ksize, strides, padding='SAME')#  + self.epsilon
        rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.max_pool_grad_v2(act, z, s, ksize, strides, padding='SAME')
        tmp = c * act
        return tf.squeeze(tmp, [3])

    def prop_gconv(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through Graph Convolutional Layers.
        All essential operations are in SCIPY.
        """
        start = time.time()
        W = self.act_weights[name][0]  # weight
        # print("prop_gconv", type(W))

        b = self.act_weights[name][1]  # bias
        # print("\nInside gconv")
        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape, type(W[0,0]))
        # activation
        N, M, Fin = activation.shape
        # N = tf.shape(activation)[0]
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = W.shape[-1]
        #print("\n\tname, inside of the prop_gconv:", name)
        # print("N, M, Fin, Fout", N, M, Fin, Fout)
        K = int(W.shape[0] / Fin)

        W = np.reshape(W, (int(W.shape[0] / K), K, Fout))
        W = np.transpose(W, (1, 0, 2))  # K x Fin x Fout

        # print("relevance_before_calculation_sum\n", relevance.numpy().sum(axis=(1, 2)))
        # print("")
        # print("W shape", W.shape)
        # print("Activation after running, shape", activation.shape)
        # print("Relevance shape", relevance.shape)
        #
        # ## !!! no need in that anymore
        # ## polynomials = self.polynomials
        # print("polynom order_K:", K)
        # print("Polynomials shape", type(polynomials))
        #
        # print("W shape before loop", W.shape)

        rel = tf.zeros(shape=[N, M * Fin]) #, dtype=np.float32)
        # polynomials[0] = polynomials[0]*W[:, :, i]
        # for i in range(1, K):

        for i in range(0, Fout):
            w = polynomials.dot(W[:, :, i])
            # print("w_pos", w_pos.shape, type(w_pos[0,0]))
            # print("i in the loop, scipy", i)
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, W[:, :, i])


            w = tf.reshape(w, [M, M, Fin])
            w = tf.transpose(w, perm=[0, 2, 1]) # perm=[0, 2, 1]) axes=[0, 2, 1]  # M x Fin x M
            w = tf.reshape(w, [M * Fin, M])
            # print("i in the loop, tf.reshape", i)
            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            # print("i in the loop, tf.reshape", i)

            z = tf.matmul(activation, w) + self.epsilon  # N x M
            z = z - tf.cast(tf.less_equal(z, tf.zeros(shape=z.shape, dtype=np.float32)),
                            dtype=np.float32) * 2 * self.epsilon
            # print("i in the loop, tf.matmul(activation, w_pos)", i)
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w))  # N x M by transpose(M * Fin, M) = N x M * Fin
            # print("i in the loop, tf.matmul(s, np.transpose(w_pos))", i)
            rel += c * activation
        end = time.time()
        # #
        rel = tf.reshape(rel, [N, M, Fin])
        print("\n\t" + name + ",", "relevance propagation time is: ", end - start)

        return rel


    def prop_gconv_first_conv_layer_general(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        # activation = np.ones_like(activation)

        # eps = 1e-10
        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w = polynomials.dot(W[:, i])
            w = tf.reshape(w, [M, M])
            # w_pos = polynomials.dot(W[:, i])
            # w_pos = tf.maximum(0.0, w_pos)
            # w_pos = tf.reshape(w_pos, [M, M])

            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            # !!!
            # general rule for relevance redistribution
            z = tf.matmul(activation, w) + self.epsilon     # N x M
            z = z - tf.cast(tf.less_equal(z, tf.zeros(shape=z.shape, dtype=np.float32)), dtype=np.float32) * 2 * self.epsilon

            # z = z - non_zero * self.eps
            print("z", np.abs(z.numpy()).min()) #, z.numpy().max())
            s = relevance[:, :, i] / z  # N x M
            print("s", s.numpy().min(), s.numpy().max())
            c = tf.matmul(s, tf.transpose(w))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel

    def prop_gconv_first_conv_layer_constraints(self, name, activation, relevance, polynomials, lb=0, hb=20):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        lb = tf.zeros(shape=[N, M], dtype=np.float32) + lb
        hb = tf.zeros(shape=[N, M], dtype=np.float32) + hb

        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w = polynomials.dot(W[:, i])
            w = tf.reshape(w, [M, M])
            w_pos = tf.maximum(0.0, w)
            w_neg = tf.minimum(0.0, w)
            w_pos = tf.reshape(w_pos, [M, M])
            w_neg = tf.reshape(w_neg, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M

            # !!!
            # Z^B rule

            z = tf.matmul(activation, w) - tf.matmul(lb, w_pos) - tf.matmul(hb, w_neg) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w))  # N x M by transpose(M * Fin, M) = N x M * Fin
            cp = tf.matmul(s, tf.transpose(w_pos))
            cn = tf.matmul(s, tf.transpose(w_neg))
            rel += c * activation - lb * cp - hb * cn
        end = time.time()

        print("\n\t" + name + ",", "Constraints, relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        # print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def prop_gconv_first_conv_layer(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w_pos = polynomials.dot(W[:, i])
            w_pos = tf.maximum(0.0, w_pos)
            w_pos = tf.reshape(w_pos, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            # !!!
            # Z^+ rule
            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = tf.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        # print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def prop_gconv_first_conv_layer_w_w_rule(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        # activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))

        # rel = tf.math.reduce_sum(relevance, axis=2)

        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            w = polynomials.dot(W[:, i])
            # w_pos = tf.maximum(0.0, w_pos)
            w = tf.reshape(w, [M, M])

            # !!!
            # w^2 rule
            w_square = w * w
            w_denominator = tf.math.reduce_sum(w_square, axis=0)
            print("w_denominator", w_denominator.shape)
            rel += np.matmul(relevance[:, :, i], w_square) / w_denominator

        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel


    def prop_gconv_first_conv_layer_old(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        # print("\tcomparison of polynomials", polynomials.shape)
        # print(polynomials)

        # print("\tcomparison of activations", activation)

        W = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        # print("\tcomparison of biases", b)

        N, M, Fin = activation.shape
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(W.shape[-1])

        # print("activation", type(activation))
        # print("weights of current gconv, w:", W.shape)

        # print("\t\t\activation", activation)
        # print("weights of current gconv, w:", w)

        K = int(W.shape[0] / Fin)

        # W = self.model._get_session().run(w)
        # B = self.model._get_session().run(b)
        # W = w

        # !!!
        # TODO: activations change
        # The relevance on the input layer does not take activations into account
        activation = np.ones_like(activation)

        # print("\trelevance as output of the first conv layer shape", relevance.shape)
        # print(relevance.numpy().sum(axis=2).sum(axis=1))


        rel = tf.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            # l_w = W[0, i] * polynomials[0]
            # lw = l_w.todense()
            # for k in range(1, K):
            #     l_w += W[k, i] * polynomials[k].todense()
            # w_pos = l_w
            #
            # w_pos = tf.sparse.sparse_dense_matmul(polynomials, tf.expand_dims(W[:, i], 1))
            # self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            # self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w_pos = polynomials.dot(W[:, i])
            w_pos = tf.maximum(0.0, w_pos)
            w_pos = tf.reshape(w_pos, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            # !!!
            # Z^+ rule
            activation = tf.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = tf.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = tf.matmul(s, tf.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        #
        # print("\n\t relevance as an input of the first conv layer shape:", rel.shape)
        # print(rel.numpy().sum(axis=1))
        # print("rel", rel.shape)
        # print("rel_after_calculation_sum", rel.numpy().sum(axis=1))
        # print("rel_after_calculation, min, max", rel.numpy().min(), rel.numpy().max())
        return rel



    # def calc_Laplace_Polynom_old(self, L, K):
    def calc_Laplace_Polynom_old(self, L, K):
        """
        Calculate Chebyshev polynoms based on Laplace matrix.
        :param L: Laplace matrix M*M
        :param K: Number of polynoms with degree from 0 to K-1
        :return: Chebyshev Polynoms in scipy.sparse.coo, shape (M*M, K)
        """
        # N, M, Fin = self.X.get_shape()
        M = int(L.shape[0])
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.

        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        polynomials = []
        if K > 1:
            # only rank 2 for sparse_tensor_dense_matmul
            T0 = scipy.sparse.identity(M, dtype=np.float32, format="csr")
            I = scipy.sparse.identity(M, dtype=np.float32, format="csr")
            T1 = L
            # polynomials.extend([I, T1]) # the first matrix is I matrix
            # polynomials = scipy.sparse.hstack([I.reshape(M * M, 1), T1.reshape(M * M, 1)])
            polynomials = [I.reshape(M * M, 1), T1.reshape(M * M, 1)]
        for k in range(2, K):
            T2 = 2 * L * T1 - T0  #
            # polynomials = scipy.sparse.hstack([polynomials, T2.reshape(M * M, 1)])
            polynomials.append(T2.reshape(M * M, 1))
            T0, T1 = T1, T2
        # print(polynomials[0].shape)
        # print(len(polynomials))
        polynomials = scipy.sparse.hstack(polynomials)
        polynomials = polynomials.tocsr()
        # print(type(polynomials))
        # polynomials = polynomials.tocoo()
        # indices = np.column_stack((polynomials.row, polynomials.col))
        # polynomials = tf.sparse.SparseTensor(indices, polynomials.data, polynomials.shape)
        # return tf.sparse.reorder(polynomials)
        return polynomials
