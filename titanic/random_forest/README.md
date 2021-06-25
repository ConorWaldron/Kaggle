# Decision Trees & Random Forests

Random Forests consisting of Decision Trees can be used for both classification and regression problems, and for data clustering/visualisation and they can handle data types that are Boolean, categorical or continuous, they can also be used if there is missing data in either the training or test sets. Decision trees aim to learn simple decision rules inferred from training data. 

Terminology

* root node - represents the entire population and poses a question that divides into 2 sets.
*	internal/decision node - a node with arrows coming from and going to it, it also contains a question to split data into 2 sets.
*	leaf/terminal node - final node which does not split, does not contain a question, just contains collection of data points that followed that path.
*	Gini impurity - leafs rarely have perfectly separated the data so that they only contain targets or only contain not targets, but rather they are usually a mix (maybe lots of targets and a few non targets). This mixture makes them impure and the quantity of separation achieved can be measured by the Gini impurity value (shown later).

Handling data types
*	Boolean data - naturally catered for with True or False question, node that splits in two.
*	Categorical data - handled with series of binary options, such as contains categories A, B, C and D, vs contains E, F, and G, then next sub internal node may be contains A and B or contains C and D and so on.
*	Numerical data is handled by splitting data at a weight value, eg is value < 80. The value of the weight is chosen by fitting data. This makes numerical data categorical which is easier to handle.

Decision tree structure
*	nodes have to be binary options.
*	trees don’t need to be symmetrical, the questions on the left branch do not need to be mirror opposite of questions on right branch).

<p align="center"><a href="../../assets/Decision_Tree_Structure.png"><img src="../../assets/Decision_Tree_Structure.png" width=800></a></p>

Building a decision tree
The order of questions in the decision tree is very important and it affects the accuracy of the model. Below we describe how to pick the order using an example where we are trying to predict a Boolean variable (heart disease), using three Boolean input variables (chest pain, blood circulation and blocked arteries).
*	pick the root node question by trying each of the input variables in turn and seeing how they will split the data into two groups. Look at the leaf nodes for each case, and assess how well this decision node split the data according to the target variable. Ideally we want this question to split the target variable perfectly, so all the people with heart disease go to one leaf and all those without go to the other. In reality you don’t get perfect separation so you quantify how well the decision split the data using the Gini impurity measure for each leaf, then calculated the weighted Gini impurity from all leaves of the decision node. Repeat process for all input variables and select the one that results in the lowest Gini impurity value. Note record the Gini impurity value for each leaf and node as that will be used later.
*	If there are input variables that have not been used yet, pick sub nodes for each leaf node remaining. This is a repeat process as above. Calculate the gini impurity for each leaf node and check that the impurity is improving (reducing) due to the addition of the extra decision variable.
*	Repeat until all input variables are used or stop when adding additional data does not reduce the Gini impurity value of the leaves compared to the parent node.
*	If you were dealing with numerical data, there are more options as there are different ways to split the variable into discrete categorical variables. Hence this process becomes a lot more work.

<p align="center"><a href="../../assets/Building_Decision_tree.png"><img src="../../assets/Building_Decision_tree.png" width=800></a></p>


## Random Forests
Decision trees are easy to build, easy to use and interpretable, but they are inaccurate when working with data not used in the training set. But, Random Forests overcome this problem and increase the accuracy of the model.

A random forest consists of
*	hundreds of decision trees (it is an ensemble model)
*	the decision trees are all made using different bootstrapped data set, (a dataset where you pick with replacement from your original data set) hence they will be slightly different.
*	Each decision tree is built using only a random subset of variables/features at each step, hence your hundreds of decision tress in the forest will have a wide diversity (this diversity makes a forest more accurate than an individual tree). Typically if you have N variables, you should consider the square root of N randomly chosen variables at each step in your decision tree. Note you are allowed to use each variable many times, so just because variable X was used in the 2nd layer of the tree, that doesn’t mean you cant use X again at 9th layer in the tree.
*	You then run every data point, through each tree in your random forest. You then compare the predictions from all your trees and you chose the most common output from your forest to be the output from the tree. Its like all the trees in the forest vote on the output. Terminology, getting this aggregate value is called bagging.
*	Terminology, the test data is the data not used in the bootstrapped data set, this is called an out of bag dataset.
*	The random forest creation process is often repeated 10 times, using  different hyper parameters when building the random forest, for example the number of variables you consider at each step when building a decision tree (usually somewhere around the square root of N, where N is the number of features), or a different number of trees in the forest, or a different fraction for the bootstrap sample size (if 100 rows in original dataset, does your bootstrap sample consists of 50, 90, 100 or 120 rows randomly chosen with replacement), or maximum tree depth.

<p align="center"><a href="../../assets/Building_Random_Forest.png"><img src="../../assets/Building_Random_Forest.png" width=800></a></p>
