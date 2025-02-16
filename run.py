import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据集路径
data_dir = r"D:/job_hunting/Comprehensive training/Traffic-Light-Classifier-master/traffic_light_images"
class_names = ['red', 'yellow', 'green']
batch_size = 32

# 加载数据集
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'training'),
    seed=123,
    image_size=(32, 32),
    batch_size=batch_size
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'test'),
    seed=123,
    image_size=(32, 32),
    batch_size=batch_size
)

# 数据集大小
train_dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
test_dataset_size = tf.data.experimental.cardinality(test_dataset).numpy()

# 数据预处理
def preprocess_image(image, label):
    image = tf.image.resize(image, (32, 32))
    image = image / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess_image)
test_dataset = test_dataset.map(preprocess_image)

# 创建LeNet模型
class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(3)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 评估函数
def evaluate(dataset):
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in dataset:
        logits = model(x, training=False)
        accuracy.update_state(y, logits)
    return accuracy.result().numpy()

# 创建损失函数和优化器
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 创建模型
model = LeNet()

# 创建评估器
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# 定义训练步骤
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_object(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy.update_state(y, logits)

# 创建会话并训练模型
epochs = 10

train_acc_list = []
val_acc_list = []

for epoch in range(epochs):
    train_dataset = train_dataset.shuffle(train_dataset_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for step, (x, y) in enumerate(train_dataset):
        train_step(x, y)

    train_accuracy = accuracy.result().numpy()
    accuracy.reset_states()

    for x, y in test_dataset:
        logits = model(x, training=False)
        accuracy.update_state(y, logits)

    validation_accuracy = accuracy.result().numpy()
    accuracy.reset_states()

    train_acc_list.append(train_accuracy)
    val_acc_list.append(validation_accuracy)

    print("Epoch {} ...".format(epoch + 1))
    print("Training Accuracy = {:.3f}".format(train_accuracy))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print()

# 可视化训练结果
plt.plot(range(epochs), train_acc_list, label='Training Accuracy')
plt.plot(range(epochs), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 在测试集上评估模型
for x, y in test_dataset:
    logits = model(x, training=False)
    accuracy.update_state(y, logits)

test_accuracy = accuracy.result().numpy()
print("Test Accuracy = {:.3f}".format(test_accuracy))

# 保存模型
model.save_weights('model/traffic_light_model.ckpt')
print("Model saved")
