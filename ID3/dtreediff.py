import math 
import DiffPrivacy as df



#This computes the freq of each distinct value in lst(target attr:) 
@df.privatize
def Compute_Freq(**kwargs):
    val_freq = {}
    lst = kwargs['input'] 
    for val in lst:
        if (val_freq.has_key(val)):
            val_freq[val] += 1.0
        else:
            val_freq[val] = 1.0

    return val_freq    


def entropy(data, target_attr):
  
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    target_values = [record[target_attr] for record in data]
    kwargs = {'input': target_values}

    val_freq = Compute_Freq(**kwargs)    
    # Calculate the entropy of the data for the target attribute
    len_data = sum([freq for freq in val_freq.values()])

    for freq in val_freq.values():
        data_entropy += (-freq/len_data) * math.log(freq/len_data, 2) 
      

    return data_entropy
  
#to compute the IG  
def gain(data, attr, target_attr):
   
    val_freq = {}
    subset_entropy = 0.0


    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records 
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    return (entropy(data, target_attr) - subset_entropy)


#This was just to test
def split_info(data, attr):
    
    split_inf = 0.0
    values = [rec[attr] for rec in data]
    distinct = set(values)
    
    for val in distinct:
        frac = float(values.count(val))/len(values)
    	split_inf  +=  -1* frac* math.log(frac, 2)
    return split_inf 



def most_frequent(lst): 
    
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    kwargs = {'input':lst}
    frequencies = Compute_Freq(**kwargs)

    for k,v in frequencies.items():        
        if v > highest_freq:
            most_freq = k
            highest_freq = v
            
    return most_freq, highest_freq


def unique(lst):
    
    lst = lst[:]
    unique_lst = []


    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            

    return unique_lst


#chooses an attribute with highest IG
def choose_attribute(data, attributes, target_attr):
   
    data = data[:]
    best_gain = -999999.999999
    best_attr = None
   

    for attr in attributes:
        curr_gain =  gain(data, attr, target_attr)
        if (curr_gain >= best_gain and attr != target_attr):
  
            best_gain = curr_gain
            best_attr = attr
            
    return best_attr


#Returns a list of all the records in <data> with the value of an attribute matching the given value.

def get_examples(data, attr, value):
    
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst

#computes and returns the classsification 
def get_classification(record, tree):
   
    
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        if record[attr] not in tree[attr]:
        	return None

        t = tree[attr][record[attr]]
        
        return get_classification(record, t)

def classify(tree, data):
   
    data = data[:]
    classification = []
    
    
    for record in data:
        classification.append(get_classification(record, tree))

    return classification


@df.privatize
def Length(**kwargs):
	return len(kwargs['input'])

def create_decision_tree(data, attrbs, target_attr):
    
    data = data[:]
    vals = [record[target_attr] for record in data]
    default, freq = most_frequent([record[target_attr] for record in data])

    kwargs  = {'input':vals}  
    len_vals = Length(**kwargs)

    # checking the attributes list for emptiness
    if not data or (len(attrbs) - 1) <= 0:
        return default

    elif freq >= len_vals:  
        return default 
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attrbs, target_attr)

        # Create a new decision tree/node with the best attribute only 
        tree = {best:{}}

        # Create a new decision tree/sub-node for each of the distinct values in the best attribute field
        for val in unique([record[best] for record in data]):
           
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attrbs if attr != best],target_attr) 

            # Add the new subtree to the empty dictionary object in the new tree/node.
            tree[best][val] = subtree

    return tree
