# Siamese Network with Tensorflow for Swiss Roll Data
# Authors: Hamid Karimi and Harrison LeFrios
# Michigan State University


from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
#from DataShuffler import *
from sklearn import metrics
from sklearn.cluster import KMeans
import sys

def generate_data(N):
    np.random.seed(0)
    X = np.loadtxt("swissroll.dat")
    Y = np.zeros((X.shape[0], 1))
    X1 = X[0:400, :]
    X2 = X[400:800, :]
    X3 = X[800:1200, :]
    X4 = X[1200:1600, :]
    Y[0:400] = 1
    Y[400:800] = 2
    Y[800:1200] = 3
    Y[1200:1600] = 4
    y1 = Y[0:400, :]
    y2 = Y[400:800, :]
    y3 = Y[800:1200, :]
    y4 = Y[1200:1600, :]

    T = np.concatenate((X1[320:400, :],X2[320:400, :],X3[320:400, :],X4[320:400, :]))
    T_y = np.concatenate((y1[320:400, :],y2[320:400, :],y3[320:400, :],y4[320:400, :]))
    X = np.concatenate((X1[0:320,:], X2[0:320,:], X3[0:320,:], X4[0:320,:]))
    Y = np.concatenate((y1[0:320,:], y2[0:320,:], y3[0:320,:], y4[0:320,:]))

    X1 = X1[0:320, :]
    X2 = X2[0:320, :]
    X3 = X3[0:320, :]
    X4 = X4[0:320, :]
    data = {'X':X, 'Y':Y, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'n1': X1.shape[0], 'n2': X2.shape[0],
            'n3': X3.shape[0], 'n4': X4.shape[0], 'T':T, 'T_y':T_y}
    #plt.plot(X1[:, 0], X1[:, 1], 'rs', X2[:, 0], X2[:, 1],'go', X3[:, 0], X3[:, 1], 'bd', X4[:, 0], X4[:, 1], 'm*',markersize=10)
    #plt.show()

    return X, Y, T, T_y, data

# Parameters
learning_rate = 0.000005
training_epochs = 200
batch_size = 1
display_step = 1

# Network Parameters
n_hidden = 25
n_hidden_1 = n_hidden  # 1st layer number of features
n_hidden_2 = n_hidden  # 2nd layer number of features
n_hidden_3 = n_hidden  # 3rd layer number of features
n_input = 3  # Swiss Roll data input (img shape: 3 dimension)
#n_classes = 2  # Swiss Roll total classes (4 class)
lambda_param = 0.5  # trade-off between push and pull
output_dim = 2 #  dimensionality reduction
optimization_option = 2  # different type of option

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])  # batch size, input dimension

same_class_data = tf.placeholder(tf.float32, [None, n_input])
diff_class_data = tf.placeholder(tf.float32, [None, n_input])


def compute_euclidean_norm(v1, v2):  # Computes the euclidean distance between two tensorflow variables
    d = tf.reduce_sum(tf.square(tf.sub(v1, v2)), 0)
    return d


def compute_euclidean_norm_np(v1, v2):  # Computes the euclidean distance between two tensorflow variables
    d = np.sum(np.square(np.subtract(v1, v2)), axis=0)
    return d

def multilayer_perceptron(x, weights, biases):  # Create model
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])  # Hidden layer 1
    layer_1 = tf.nn.relu(layer_1)  # RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])  # Hidden layer 2
    # layer_2 = tf.nn.relu(layer_2)  # RELU activation
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])  # Hidden layer 3
    # layer_3 = tf.nn.relu(layer_3)  # RELU activation
    # out_layer = tf.matmul(layer_3, weights['out']) + biases['out']  # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']  # Output layer with linear activation
    return out_layer

# Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
#     'out': tf.Variable(tf.random_normal([n_hidden_3, output_dim]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'b3': tf.Variable(tf.random_normal([n_hidden_3])),
#     'out': tf.Variable(tf.random_normal([output_dim]))
# }

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, output_dim]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([output_dim]))
}


def next_batch(data, i, size):
    #  X, Y = data['X'], data['Y']
    data_x = data['X'][i*size:i*size+size, :]
    data_y = data['Y'][i * size:i * size + size, :]
    return data_x, data_y


def contrastive_loss(v, left, right, left_c, right_c):
    # embed space
    d1 = compute_euclidean_norm(v, left)  # euclidean distance of similar class in embed space
    d2 = compute_euclidean_norm(v, right)  # euclidean distance of different class in embed space
    d3 = compute_euclidean_norm(v, left_c)  # distance between main data and centroid
    d4 = compute_euclidean_norm(left, left_c)  # distance between left and centroid
    d5 = compute_euclidean_norm(right, right_c)  # distance between right and centroid
    d6 = compute_euclidean_norm(left_c, right_c)  # distance between two centroids

    if optimization_option == 1:  # LMNN
        d2 = tf.sub(d1, d2)  # difference between similar and dissimilar
        d2 = tf.maximum(tf.add(tf.constant(1.0), d2), tf.constant(0.0))  # margin between simmilar and dissimilar distance
        loss = tf.add(tf.mul(lambda_param, d1), tf.mul(tf.sub(tf.constant(1.0), lambda_param), d2))
    elif optimization_option == 2:  # margin + push of centroids
        d2 = tf.sub(d1, d2)  # difference between similar and dissimilar
        d2 = tf.maximum(tf.add(tf.constant(1.0), d2), tf.constant(0.0))  # margin between simmilar and dissimilar distance
        d6 = tf.sub(tf.constant(1.0), d6)
        loss = tf.add(d1, d2)
        loss = tf.add(loss, d6)
    elif optimization_option == 3:  # All combined
        d2 = tf.sub(d1, d2)  # difference between similar and dissimilar
        d2 = tf.maximum(tf.add(tf.constant(1.0), d2), tf.constant(0.0))  # margin between simmilar and dissimilar distance
        d6 = tf.maximum(tf.sub(tf.constant(1.0), d6), tf.constant(0.0))  # margin between clusters
        loss1 = tf.add(d1, d2)
        loss2 = tf.add(d3, d4)
        loss3 = tf.add(d5, d6)
        loss = tf.add(loss1, loss2)
        loss = tf.add(loss, loss3)
        loss = tf.div(loss, tf.constant(6.0))
    elif optimization_option == 4:
        d2 = tf.sub(d1, d2)  # difference between similar and dissimilar
        d2 = tf.maximum(tf.add(tf.constant(1.0), d2), tf.constant(0.0))  # margin between simmilar and dissimilar distance
        d3 = tf.maximum(tf.sub(d3, tf.constant(1.0)), tf.constant(0.0))  # margin w.r.t. centroids
        d4 = tf.maximum(tf.sub(d4, tf.constant(1.0)), tf.constant(0.0))  # margin w.r.t. centroids
        d5 = tf.maximum(tf.sub(d5, tf.constant(1.0)), tf.constant(0.0))  # margin w.r.t. centroids
        d6 = tf.maximum(tf.sub(tf.constant(1.0), d6), tf.constant(0.0))  # margin between clusters
        loss1 = tf.add(d1, d2)
        loss2 = tf.add(d3, d4)
        loss3 = tf.add(d5, d6)
        loss = tf.add(loss1, loss2)
        loss = tf.add(loss, loss3)
        #loss = tf.div(loss, tf.constant(6.0))
    elif optimization_option == 5:
        d2 = tf.sub(d1, d2)  # difference between similar and dissimilar
        d2 = tf.maximum(tf.add(tf.constant(1.0), d2), tf.constant(0.0))  # margin between simmilar and dissimilar distance
        loss1 = tf.add(d1, d2)
        #  distance between clusters should be less than sum of distances
        loss2 = tf.maximum(tf.sub(tf.add(d3, d4), d6), tf.constant(0.0))
        loss = tf.add(loss1, loss2)

    return loss


def find_left(v):
    data_embed = multilayer_perceptron(same_class_data, weights, biases)
    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(v, data_embed)), reduction_indices=1))
    idx = tf.argmax(l2diff, axis=0)
    idx = tf.cast(idx, "int32")
    return data_embed[idx]


def find_right(v):
    data_embed = multilayer_perceptron(diff_class_data, weights, biases)
    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(v, data_embed)), reduction_indices=1))
    idx = tf.argmin(l2diff, axis=0)
    idx = tf.cast(idx, "int32")
    return data_embed[idx]


def find_left_centroid():
    data_embed = multilayer_perceptron(same_class_data, weights, biases)
    center = tf.reduce_mean(data_embed, 0)#, keep_dims=True)
    return center


def find_right_centroid():
    data_embed = multilayer_perceptron(diff_class_data, weights, biases)
    center = tf.reduce_mean(data_embed, 0)#, keep_dims=True)
    return center

y = multilayer_perceptron(x, weights, biases)  # embed space
left_embed = find_left(y)  # embed space
right_embed = find_right(y)  # embed space
left_centroid = find_left_centroid()  # embed space
right_centroid = find_right_centroid()  # embed space
cost = tf.reduce_mean(contrastive_loss(y, left_embed, right_embed, left_centroid, right_centroid))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# v1 = tf.placeholder(tf.float32, [None, 2])
# v2 = tf.placeholder(tf.float32, [None, 2])
# euclid = compute_euclidean_norm(v1, v2)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()


def KmeansClustering(dataMatrix, gTruthlabels, numClusters = None, Ninit = 20):
    if numClusters == None:
        numClusters = len(set(gTruthlabels))
    km = KMeans(init='k-means++', n_clusters=numClusters, n_init=Ninit).fit(dataMatrix)
    labels = km.labels_
    numClusters = len(set(labels)) - (1 if -1 in labels else 0)
    homogeneity = metrics.homogeneity_score(gTruthlabels, labels)
    completeness = metrics.completeness_score(gTruthlabels, labels)
    v_measure = metrics.v_measure_score(gTruthlabels, labels)
    adjusted_rand = metrics.adjusted_rand_score(gTruthlabels, labels)
    adjusted_mutual_info = metrics.adjusted_mutual_info_score(gTruthlabels, labels)
    silhouette = metrics.silhouette_score(dataMatrix, labels)
    Metrics = {'method': 'Kmeans', 'numClusters': numClusters, 'homogeneity' :homogeneity, 'completeness': completeness,
               'v_measure': v_measure, 'adjusted_rand': adjusted_rand, 'adjusted_mutual_info': adjusted_mutual_info, 'silhouette': silhouette}

    return Metrics


def printClusteringMetrcis(metrics, logFile = None):
    print (80 * '=')
    print ("Method: ", metrics['method'])
    print('Estimated number of clusters: %d' % metrics['numClusters'])
    print("Homogeneity: %0.6f" % metrics['homogeneity'])
    print("Completeness: %0.6f" % metrics['completeness'])
    print("V-measure: %0.6f" % metrics['v_measure'])
    print("Adjusted Rand Index: %0.6f" % metrics['adjusted_rand'] )
    print("Adjusted Mutual Information: %0.6f" % metrics['adjusted_mutual_info'])
    print("Silhouette Coefficient: %0.6f" % metrics['silhouette'])
    print(80 * '=')


def testing_accuracy(z):
    n1 = len(z)//4
    Y1 = np.zeros((n1)) + 1
    Y2 = np.zeros((n1)) + 2
    Y3 = np.zeros((n1)) + 3
    Y4 = np.zeros((n1)) + 4
    Y = np.concatenate((Y1, Y2, Y3, Y4))
    YL2 = Y.tolist()
    KmeansMertics = KmeansClustering(z, YL2)
    return KmeansMertics


def plot_embedding(z1, z2, z3, z4):
    plt.plot(z1[:, 0], z1[:, 1], 'rs', markersize=10)
    plt.plot(z2[:, 0], z2[:, 1], 'go', markersize=10)
    plt.plot(z3[:, 0], z3[:, 1], 'bd', markersize=10)
    plt.plot(z4[:, 0], z4[:, 1], 'm*', markersize=10)
    plt.show()


# Launch the graph
sess = tf.Session()
with sess.as_default():
    sess.run(init)
    N = 1600
    X, Y, T, T_y, data = generate_data(N)#X,Y = training, T,T_y = test
    n = Y.shape[0]
    lb = preprocessing.LabelBinarizer()
    lb.fit(Y) #LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        c = 0.0
        min_cost = sys.maxsize
        total_batch = int(n/batch_size)#int(mnist.train.num_examples/batch_size)
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        np.random.seed(0)
        tf.set_random_seed(0)
        if epoch > 100:
             optimization_option = 1
        # elif epoch >150:
        #     optimization_option = 3

        for i in range(total_batch):
            batch_x, label = next_batch(data, i, batch_size)
            #batch_y = lb.transform(batch_y)
            cls = int(label)
            #str1 =  'X' + str(cls)
            #cls = tf.cast(label, "int64")
            #str = tf.concat(0, 'X' + str(cls))
            same_data = data['X' + str(cls)]
            #same_data = np.delete(same_data, i, 0)  # delete current x, here it should be batch of x
            cls2 = np.random.randint(1, 5, 1)
            if cls2 == cls:
                cls2 = (cls2 + 1)
                if cls2 == 5:
                    cls2 = 1
            cls2 = int(cls2)
            diff_data = data['X' + str(cls2)]

            #distances = [sess.run(euclid_dist, feed_dict={v1: batch_x, v2: same_data[e, :]}) for e in range(0,same_data.shape[0])]
            #_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            #y2 = sess.run(y, feed_dict={x: batch_x})
            #[y_left, y_right, c_left, c_right] = sess.run([left_embed, right_embed, left_centroid, right_centroid], feed_dict={x: batch_x, same_class_data: same_data,
            #                                              diff_class_data: diff_data})
            #print("left = ", y_left, ", right = ", y_right, "left centroid = ", c_left, ", right centroid = ", c_right)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, same_class_data: same_data, diff_class_data: diff_data})
            # Compute average loss
            #d1 = compute_euclidean_norm_np(y_left, c_left)
            #d2 = compute_euclidean_norm_np(c_left, c_right)
            #print("d1 = ", d1, ", d2 = ", d2)

            if c < min_cost:
                min_cost = c
            #avg_cost += c / total_batch
            #print("i = ", i, "c = ", c,"\n")
        # Display logs per epoch step
        #if epoch % display_step == 0:
        #batch_y = lb.transform(T_y)
        z_test = sess.run(y, feed_dict={x: T})
        z_train = sess.run(y, feed_dict={x: X})
        KmeansMertics_test = testing_accuracy(z_test)
        KmeansMertics_train = testing_accuracy(z_train)
        #test_cost = sess.run(cost, feed_dict={x: T, y: batch_y})
        #print("Epoch:", (epoch+1), ", Training cost=", "{:.9f}".format(avg_cost), ", Test Cost = ", "{:.9f}".format(test_cost))
        print("Epoch: %d, homogeneity = %f, completeness = %f, v_measure = %f, train_v_measure = %f" % ((epoch+1), KmeansMertics_test['homogeneity'],
                                                                                  KmeansMertics_test['completeness'],
                                                                                  KmeansMertics_test['v_measure'], KmeansMertics_train['v_measure']))
        #print("Epoch:", (epoch + 1), ", Min cost=", "{:.9f}".format(min_cost))

        if epoch % 50 == 0:
            C1 = sess.run(y, feed_dict={x: data['X1']})
            C2 = sess.run(y, feed_dict={x: data['X2']})
            C3 = sess.run(y, feed_dict={x: data['X3']})
            C4 = sess.run(y, feed_dict={x: data['X4']})
            plot_embedding(C1, C2, C3, C4)
    print("Optimization Finished!")
    C1 = sess.run(y, feed_dict={x: data['X1']})
    C2 = sess.run(y, feed_dict={x: data['X2']})
    C3 = sess.run(y, feed_dict={x: data['X3']})
    C4 = sess.run(y, feed_dict={x: data['X4']})
    plot_embedding(C1, C2, C3, C4)
    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
