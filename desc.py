import numpy as np
import pandas as pd

df=pd.read_csv('Iris.csv')
df2=pd.read_csv('Iris.csv')

train=df.drop('Id', axis = 1)
train=train.replace('Iris-setosa',0).replace('Iris-versicolor',1).replace('Iris-virginica',2)

train2=df.drop('Id', axis = 1)
"""if train2['Species']=='Iris-setosa':
    train2["idi"]=0
elif train2['Species']=='Iris-versicolor':
    train2["idi"]=1
elif train2['Species']=='Iris-verginica':
    train2["idi"]=1"""

def purity(data):
    label=data[:,-1]
    #print(label)
    #print(np.unique(label))
    if len(np.unique(label))==1:
        return True
    else:
        return False
#print(purity(train[train.PetalWidthCm<0.8].values))

def classify(data):
    classification=""
    column=data[:,-1]
    print("clo",column)
    unique,count_unique=np.unique(column,return_counts=True)
    print("c",count_unique)
    index=np.argmax(count_unique)
    print("ind",index)
    if index==0 and len(count_unique)==3:
        classification='iris-setosa'
    elif index==0 and len(count_unique)==2:
        classification='iris-versicolor'
    elif index==1:
        classification='iris-virginica'
    elif len(count_unique)==1 and column[0]==0 and index==0:
        print("bjbj")
        classification='iris-setosa'
    elif len(count_unique)==1 and column[0]==1 and index==0:
        classification='iris-versicolor'
    elif len(count_unique)==1 and column[0]==2 and index==0:
        classification='iris-virginica'   
    
    return classification 
#print(classify(train[(train.PetalWidthCm>1.5)&(train.PetalWidthCm<2)].values))

def splits(data):
    pot_splits={}
    n_rows,n_columns=data.shape
    #print(data.shape)
    for column_index in range(n_columns-1):
        pot_splits[column_index]=[]
        values=data[:,column_index]
        unique_values=np.unique(values)
        for index in range(len(unique_values)):
            if index!=0:
                current_value=unique_values[index]
                previous_value=unique_values[index-1]
                potential=(current_value+previous_value)/2
                pot_splits[column_index].append(potential)
    return pot_splits
pot_splits=splits(train.values)
print('pot_splits',pot_splits)

def split_data(data,split_column,split_value):
    split_column_values=data[:,split_column]

    data_below=data[split_column_values<=split_value]
    data_above=data[split_column_values>split_value]

    return data_below,data_above

split_column=3
split_value=0.8

data_below,data_above=split_data(train.values,split_column,split_value)
#print(data_above)

def calc_entropy(data):
    column=data[:,-1]
    _,counts=np.unique(column,return_counts=True)

    probabilities=counts/counts.sum()
    entropy=sum(probabilities*-np.log2(probabilities))

    return entropy
##print(calc_entropy(data_above))

def calc_overall_entropy(data_below,data_above):
    n_data_points=len(data_below)+len(data_above)

    p_data_below=len(data_below)/n_data_points
    p_data_above=len(data_above)/n_data_points

    overall_entropy=(p_data_below*calc_entropy(data_below))+(p_data_above*calc_entropy(data_above))
    return overall_entropy

#print(calc_overall_entropy(data_below,data_above))

def determine_best_split(data,pot_splits):
    overall_entropy=999
    for column_index in pot_splits:
        for value in pot_splits[column_index]:
            data_below,data_above=split_data(train.values,split_column=column_index,split_value=value)
            current_overall_entropy=calc_overall_entropy(data_below,data_above)

            if current_overall_entropy<=overall_entropy:
                overall_entropy=current_overall_entropy
                best_split_column=column_index
                best_split_value=value
    return best_split_column,best_split_value
#print(determine_best_split(train.values,pot_splits))

def decision_tree_algorithm(df,counter=0):
    if counter==0:
        global column_headers
        column_headers=df.columns
        data=df.values
    else:
        data=df
    
    #base case
    if purity(data):
        classification=classify(data)
        print("class",classification)
        return classification
    
    #recursive part
    else:
        counter+=1

        #main functions
        pot_splits=splits(data)
        split_column,split_value=determine_best_split(data,pot_splits)
        print('split-col',split_column)
        data_below,data_above=split_data(data,split_column,split_value)

        #making sub-tree
        feature_name=column_headers[split_column]
        question="{} <= {}".format(feature_name,split_value)
        sub_tree={question:[]}

        yes_answer=decision_tree_algorithm(data_below,counter)
        print("yes",yes_answer)
        no_answer=decision_tree_algorithm(data_above,counter)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

        return sub_tree
    
tree=decision_tree_algorithm(train)
print(tree)

test=pd.read_csv('test.csv')

def classify_example(example,tree):
    question=list(tree.keys())[0]
    feature_name,comparision_operator,value=question.split()

    #ask question

    if example[feature_name]<=float(value):
        answer=tree[question][0]
        #print('ghjh',float(value))
    else:
        answer=tree[question][1]
    #base case
    if not isinstance(answer,dict):
        return answer
    else:
        residual_tree=answer
        return classify_example(example,residual_tree)
example=test.iloc[2]
print('ghh',classify_example(example,tree))
#print('pot_splits',pot_splits)

'''def calc_accuracy(df,tree):
    df['classification']=df.apply(classify_example,axis=1,args=(tree,))
    print('dd',df['classification'][2])
    """if df['classification'][0]==0:
        print("s")"""
    df['classification_correct']=df.classification==df2.Species
    accuracy=df.classification_correct.mean()
    return accuracy
print('sf',calc_accuracy(test,tree))'''




