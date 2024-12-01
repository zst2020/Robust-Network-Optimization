from tensorflow.keras.layers import Input, Reshape, Dense, Flatten, Dropout, SimpleRNN, LSTM
from tensorflow.keras.layers import Activation, InputLayer
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

LR = 0.001
TREE_DEPTH = 7


def DNN_SE(dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mean_absolute_percentage_error')
    modeldir = "figs/" + 'DeepNeuralNetwork' + '.png'
    # tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)
    return model


class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate=1.0, num_classes=1):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        # Apply the feature mask to the input features.
        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                              :, begin_idx:end_idx, :
                              ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = keras.activations.linear(self.pi)  # [num_leaves, num_classes]
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs


class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate=1.0, num_classes=1):
        super(NeuralDecisionForest, self).__init__()
        self.ensemble = []
        self.num_classes = num_classes
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, self.num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs


def DNN_SE_RF(dim):
    inputs = tf.keras.Input(shape=(dim,))
    num_features = 32
    num_treeout = 32
    num_trees = 3
    y = layers.Dense(32, activation="relu")(inputs)
    y = layers.Dense(num_features, activation="relu")(y)
    features = y
    tree = NeuralDecisionForest(num_trees, TREE_DEPTH, num_features, 1.0, num_treeout)
    y = tree(features)
    y = layers.Dense(16, activation="relu")(y)
    y = layers.Dense(8, activation="relu")(y)
    y = layers.Dense(4, activation="relu")(y)
    outputs = layers.Dense(1, activation="linear")(y)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mean_absolute_percentage_error')
    modeldir = "figs/" + 'NeuralDecisionForest' + '.png'
    # tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)
    return model


def create_class_mlp(dim):
    input = keras.Input(shape=(dim,))
    model = Sequential()
    Cla = layers.Dense(32, activation="relu")(input)
    Cla = layers.Dense(16, activation="relu")(Cla)
    Cla = layers.Dense(8, activation="relu")(Cla)
    Cla = layers.Dense(4, activation="relu")(Cla)
    # Cla = layers.Dense(2, activation="relu")(Cla)
    outputs = layers.Dense(2, activation="softmax")(Cla)  # 输出在0-1之间
    model = keras.Model(inputs=input, outputs=outputs)
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    return model


def se_predict_mlp(dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mse',metrics='mape')
    modeldir = "figs/" + 'SENet' + '.png'
    tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)
    return model


def class_predict_mlp(dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    modeldir = "figs/" + 'ClassNet' + '.png'
    # tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)
    return model


def rnn(input_shape, time_step):
    model = Sequential()
    model.add(LSTM(32, input_shape=(time_step, input_shape[-1]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mean_absolute_percentage_error')
    modeldir = "figs/" + 'LSTM' + '.png'
    tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)

    return model

def mlp(dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mape')
    modeldir = "figs/" + 'mlp' + '.png'
    tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)
    return model

def resnet(dim):
    input = keras.Input(shape=(dim,))
    model = Sequential()
    x1 = layers.Dense(32, activation="relu")(input)
    x2 = layers.Dense(32, activation="relu")(x1)
    x2 = layers.Dense(32, activation="relu")(x2)
    x3 = layers.Dense(32)(x2)
    x = layers.Add()([x1, x3])

    # x1 = layers.Dense(32, activation="relu")(layers.ReLU()(x))
    # x2 = layers.Dense(32, activation="relu")(x1)
    # x2 = layers.Dense(32, activation="relu")(x2)
    # x3 = layers.Dense(32)(x2)
    # x = layers.Add()([x1, x3])

    # x1 = layers.Dense(32, activation="relu")(layers.ReLU()(x))
    # x2 = layers.Dense(32, activation="relu")(x1)
    # x2 = layers.Dense(32, activation="relu")(x2)
    # x3 = layers.Dense(32)(x2)
    # x = layers.Add()([x1, x3])

    x1 = layers.Dense(32, activation="relu")(layers.ReLU()(x))
    x2 = layers.Dense(32, activation="relu")(x1)
    x2 = layers.Dense(32, activation="relu")(x2)
    x3 = layers.Dense(32)(x2)
    x = layers.Add()([x1, x3])

    x7 = layers.Dense(16, activation="relu")(layers.ReLU()(x))
    x8 = layers.Dense(8, activation="relu")(x7)

    outputs = layers.Dense(1, activation='linear')(x8)
    model = keras.Model(inputs=input, outputs=outputs)

    model.summary()
    adam = Adam(lr=LR)
    model.compile(optimizer=adam, loss='mse',metrics='mape')
    modeldir = "figs/" + 'resnet' + '.png'
    tf.keras.utils.plot_model(model, to_file=modeldir, show_shapes=True)
    return model


if __name__ == '__main__':
    # model = rnn((200,20,22),20)

    model = resnet(24)

    pass
