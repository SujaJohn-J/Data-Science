# Importing libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

spark = SparkSession.builder.appName("Diamonds").getOrCreate()

# Loading data
data = spark.read.csv("/content/drive/MyDrive/Diamonds_Pyspark/diamonds.csv", header=True, inferSchema=True)

# Data schema
data.printSchema()

# Summary statistics
data.describe().show()

# Checking null values
data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()

# Scatter plot between carat and price
sns.scatterplot(data=data.toPandas(), x="carat", y="price", hue="cut")
plt.show()

# Count plot of cut
sns.countplot(data=data.toPandas(), x="cut")
plt.show()

# Boxplot of price by cut
sns.boxplot(data=data.toPandas(), x="cut", y="price")
plt.show()

# Indexing string columns
indexer = StringIndexer(inputCols=["cut", "color", "clarity"], outputCols=["cut_index", "color_index", "clarity_index"])

# VectorAssembler
assembler = VectorAssembler(inputCols=["carat", "cut_index", "color_index", "clarity_index", "depth", "table", "'x'", "'y'", "'z'"], outputCol="features")

# Pipeline
pipeline = Pipeline(stages=[indexer, assembler])

# Preprocessing the data
preprocessed_data = pipeline.fit(data).transform(data)

# Splitting the data into train and test sets
(training_data, testing_data) = preprocessed_data.randomSplit([0.7, 0.3], seed=100)

# Decision Tree Classifier
dt = DecisionTreeClassifier(labelCol="cut_index", featuresCol="features")

# Training the model
dt_model = dt.fit(training_data)

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define the parameter grid to search
param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Define the evaluator to use
evaluator = MulticlassClassificationEvaluator(labelCol="cut_index", predictionCol="prediction", metricName="accuracy")

# Define the cross-validator to use
cv = CrossValidator(estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=100)

# Fit the cross-validator to the training data
cv_model = cv.fit(training_data)

# Make predictions on the testing data
predictions = cv_model.transform(testing_data)

# Evaluate the model using the evaluator
accuracy = evaluator.evaluate(predictions)
print("Accuracy = {:.2f}".format(accuracy))

# Print the classification report
from sklearn.metrics import classification_report
y_true = predictions.select(['cut_index']).collect()
y_pred = predictions.select(['prediction']).collect()
print(classification_report(y_true, y_pred))

# Random Forest Classifier
rf = RandomForestClassifier(labelCol="cut_index", featuresCol="features")

# Training the model
rf_model = rf.fit(training_data)

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define the parameter grid to search
param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Define the evaluator to use
evaluator = MulticlassClassificationEvaluator(labelCol="cut_index", predictionCol="prediction", metricName="accuracy")

# Define the cross-validator to use
cv = CrossValidator(estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=100)

# Fit the cross-validator to the training data
cv_model = cv.fit(training_data)

# Make predictions on the testing data
predictions = cv_model.transform(testing_data)

# Evaluate the model using the evaluator
accuracy = evaluator.evaluate(predictions)
print("Accuracy = {:.2f}".format(accuracy))

# Print the classification report
from sklearn.metrics import classification_report
y_true = predictions.select(['cut_index']).collect()
y_pred = predictions.select(['prediction']).collect()
print(classification_report(y_true, y_pred))

# Make predictions on the testing data
predictions = rf_model.transform(testing_data)

# Select the actual and predicted labels and put them in a dataframe
predictions_df = predictions.select("cut_index", "prediction").toPandas()

# Print the first 10 rows of the dataframe
print(predictions_df.head(50))

#print(predictions_df.head(50))

# print("Training data:")
# training_data.show()

# print("Testing data:")
# testing_data.show()

# convert the RDDs to NumPy arrays
train_features = np.array(training_data.select("features").rdd.map(lambda x: x[0]).collect())
train_labels = np.array(training_data.select("cut_index").rdd.map(lambda x: x[0]).collect())
test_features = np.array(testing_data.select("features").rdd.map(lambda x: x[0]).collect())
test_labels = np.array(testing_data.select("cut_index").rdd.map(lambda x: x[0]).collect())

# create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_features, train_labels, epochs=10, batch_size=64, validation_data=(test_features, test_labels))

print(test_labels)

# Make predictions on test data
predictions = model.predict(test_features)

# Convert predictions to integer class labels
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted labels
print(predicted_labels)

# Convert the test labels and predicted labels to a pandas dataframe
results_df = pd.DataFrame({'test_labels': test_labels, 'predicted_labels': predicted_labels})

# Convert the pandas dataframe to a Spark DataFrame
results_spark = spark.createDataFrame(results_df)

# Show the first 10 rows of the results DataFrame
results_spark.show(10)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Make predictions on the testing data
predictions = model.predict(test_features)
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted labels
print(predicted_labels)

# Generate classification report
target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
print(classification_report(test_labels, predicted_labels, target_names=target_names))

# Generate confusion matrix
conf_mat = confusion_matrix(test_labels, predicted_labels)
print(conf_mat)

# Generate heatmap
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(conf_mat, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(target_names)):
    for j in range(len(target_names)):
        ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="white")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Group the data by the test labels and predicted labels and count the number of occurrences of each combination
results_count = results_spark.groupby(['test_labels', 'predicted_labels']).count().orderBy(['test_labels', 'predicted_labels'])

# Convert the Spark DataFrame to a pandas DataFrame
results_count_df = results_count.toPandas()

# Create a bar chart to visualize the results
sns.barplot(x='test_labels', y='count', hue='predicted_labels', data=results_count_df)
plt.show()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)