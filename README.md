# Decision-tree
The given code is written in python and it performs a classification task using a decision tree classifier on the Iris dataset. Here's a breakdown of what the code does:  It imports necessary libraries/modules such as classifier, pandas, seaborn, math, random, sns (alias for seaborn), confusion_matrix from sklearn.metrics, train_test_split from sklearn.model_selection, pyplot from matplotlib, and tree from sklearn.
It reads the Iris dataset from a CSV file located at "C:/Users/PCS/Desktop/iris.csv" using pd.read_csv and assigns it to the variable df. 
It prints the first five rows of the dataset using df.head().  It prints the column names of the dataset using df.columns. 
It prints the count of each unique value in the 'Species' column using df['Species'].value_counts().  It prints information about the dataset using df.info(). 
It prints a random sample of 5 rows from the dataset using df.sample(5).  It creates two scatter plots using seaborn.FacetGrid and plt.scatter, one for 'Sepal.Length' vs. 'Sepal.Width' colored by 'Species', and another for 'Petal.Length' vs. 'Petal.Width' colored by 'Species'. The plots are displayed using plt.show().
It splits the dataset into training and testing sets using train_test_split from sklearn.model_selection with a test size of 0.3 (30% of the data). 
It prints the shapes (number of rows and columns) of the training and testing sets using train_set.shape and test_set.shape. 
It separates the features (input variables) and labels (output variable) for training and testing sets. 
It initializes a decision tree classifier with parameters such as entropy as the splitting criterion, maximum depth of 3, and maximum leaf nodes of 5 using tree.DecisionTreeClassifier.
It fits the classifier to the training data using clf.fit(x_train, y_train).  It plots the decision tree using tree.plot_tree from sklearn and plt.show(). 
It uses the trained classifier to make predictions on the testing data using clf.predict(x_test) and assigns the result to the variable prediction.  It prints the predictions made by the classifier. 
It calculates the confusion matrix between the true labels (y_test) and the predicted labels (prediction) using confusion_matrix from sklearn.metrics and assigns the result to the variable matix. 
It creates a heatmap visualization of the confusion matrix using seaborn.heatmap and plt.show().
