from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
def makeCart(x,y,var):
    clf=DecisionTreeClassifier(min_samples_leaf=6000).fit(x, y,)
    export_graphviz(clf,out_file='output\\tree.dot',feature_names=var)


