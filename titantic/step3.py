# -*- coding: utf-8 -*-

# import libaries
import os, csv, numpy, scipy, pandas, pylab, math

# specify variables
filename_train_input = 'data/train.csv'
filename_test_input = 'data/test.csv'
filename_output = 'data/pytest3c.csv'

# open csv file
if not os.path.exists(filename_train_input):
    print 'Unable to locate file: %s, abort!' % (filename_train_input)
    exit()

data_raw = pandas.read_csv(filename_train_input, quotechar='"',skipinitialspace=True)
filename_train_input = None

# handy information
print data_raw.head(3)
print type(data_raw)
print data_raw.dtypes
print data_raw.describe()

# this shows up which columns have missing data
print data_raw.info()

# get specific column
print data_raw.Age[0:10]
print data_raw['Age'].mean()
print type(data_raw.Age)

# get subset of columns
print data_raw[['Age','Pclass','Sex']]

# filter on column
print data_raw[data_raw.Age > 60]
print data_raw[data_raw.Age > 60][['Sex','Pclass','Age']]

# render histogram by decade
max_age = max(data_raw.Age)
data_raw.Age.dropna().hist(bins=max_age/10, range=(0,max_age), alpha=.5)
pylab.show()


def clean_data(obj):    
    # Ticket doesn't seem relevant?
    # Pclass is factored nicely already
    obj['Pclass'] = obj.Pclass.map(lambda x: -1 if numpy.isnan(x) else x)

    # Fare should be in ranges
    obj['FareBucket'] = obj.Fare.map(lambda x: -1 if numpy.isnan(x) or x < 0 else 0 if x < 10 else 1 if x < 20 else 2 if x < 30 else 3)
    
    # Name should be LastName only (to indicate family status)
    obj['LastName'] =  obj['Name'].str.replace(r'^(.*?),.*$','\\1')
    
    # Sex to Gender where female = 0 and male = 1
    obj['Gender'] = obj.Sex.map(lambda x: 0 if x.lower() == 'female' else 1 if x.lower() == 'male' else -1)
    
    # Age impute missing values then to Factor
    obj['AgeBucket'] = obj.Age.map(lambda x: -1 if numpy.isnan(x) else int(round(x/20)))
    age_medians = obj[['Gender','Pclass','FareBucket','AgeBucket']].groupby(['Gender','Pclass','FareBucket']).median()
    # attempt to impute missing values round to whole integer
    obj['AgeBucket'] = obj.apply(lambda x: int(x.AgeBucket) if x.AgeBucket != -1 and not numpy.isnan(x.AgeBucket) else int(age_medians['AgeBucket'][x.Gender][x.Pclass][x.FareBucket]), axis=1)
    # second pass, with less groping
    age_medians = obj[['Gender','Pclass','AgeBucket']].groupby(['Gender','Pclass']).median()
    obj['AgeBucket'] = obj.apply(lambda x: int(x.AgeBucket) if x.AgeBucket != -1 and not numpy.isnan(x.AgeBucket) else int(age_medians['AgeBucket'][x.Gender][x.Pclass]), axis=1)
    # third pass
    age_medians = obj[['Gender','AgeBucket']].groupby(['Gender']).median()
    obj['AgeBucket'] = obj.apply(lambda x: int(x.AgeBucket) if x.AgeBucket != -1 and not numpy.isnan(x.AgeBucket) else int(age_medians['AgeBucket'][x.Gender]), axis=1)

    # SibSp and Parch into a single FamilySize
    obj['FamilySize'] = obj['SibSp'] + obj['Parch']
    obj['FamilySize'] = obj.FamilySize.map(lambda x: x if x < 3 else 4)
    
    # Cabin should be just the alphabetic character
    obj['CabinLevel'] = obj['Cabin'].str.replace(r'^([a-zA-Z]).*$','\\1')
    obj['CabinLevel'] = obj.CabinLevel.map({numpy.nan:-1,'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7}).astype(int)
    cabin_medians = obj[['Gender','FareBucket','Pclass','CabinLevel']].groupby(['Gender','Pclass','FareBucket']).median()
    obj['CabinLevel'] = obj.apply(lambda x: int(x.CabinLevel) if x.CabinLevel != -1 and not numpy.isnan(x.CabinLevel) else int(cabin_medians['CabinLevel'][x.Gender][x.Pclass][x.FareBucket]), axis=1)
    
    # Embark impute missing values then to Factor
    obj['EmbarkedFactor'] = obj.Embarked.map({numpy.nan:-1,'S':0,'C':1,'Q':2}).astype(int)
    
    # remove unused columns
    return obj.drop(['Name','Age','Sex','Ticket','Cabin','Embarked','SibSp','Parch','Fare'],axis=1)    
    

# split into train/test sets
numpy.random.seed(777)
split_vector = numpy.random.binomial(1,.75,data_raw.shape[0]) == 1
data_train = clean_data(data_raw[split_vector])
data_test = clean_data(data_raw[-split_vector])
print 'Split set is Train: %d, Test: %d' % (data_train.shape[0], data_test.shape[0])

# cleanup data into tidy data set
#TODO:
'''
def assign_highest_continuous_attribute_information_gain(obj,column,baseline,ig=.5):
    length = len(obj)
    if length <= 1:
        return obj
    baseline_entropy = entropy(obj[baseline])
    counts = scipy.stats.itemfreq(obj[column])
    split = None
    evaluation = []
    name = "%sFactor" % (column)
    for i in counts:
        split = i[0]
        if numpy.isnan(split):
            continue
        evaluation.append( (
            split,
            baseline_entropy - entropy([1 if x <= split else 0 for x in obj[column]]),
            len(obj[obj[column] <= split]) / float(len(obj[column]))
        ) )
    return evaluation

    # select highest point that gives us more than .5 bit of information
    split = max([x for x in evaluation if evaluation[x] > ig])
    print "Splitting continous attribute: %s on %f" % (column,split)
    obj[name] = obj[column] <= split
age_split = find_highest_ig(data_train,'Age','Survived',.5)

# get binary split information gain and population coverage
a = assign_highest_continuous_attribute_information_gain(data_train,'Fare','Survived')
# sort by information gain
s = sorted(a,key=lambda x: x[1])
[x for x in s if x[1] > .5]
'''



# calculating totals

# calculate entropy for given column
# value = entropy(data['column_name'])
def entropy(obj):
    # get total number of observations in data set
    n_labels = len(obj)
    if n_labels <= 1:
        return 0
    # get array of counts for each option in this column
    #counts = numpy.bincount(obj)
    counts = scipy.stats.itemfreq(obj)
    # calculate probability for each option
    probs = counts[:,1] / float(n_labels)
    # only evaluate non-zero probabilities
    n_classes = numpy.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    # calculate entropy for given column
    ent = 0.0
    for i in probs:
        # verify we have a positive probability
        if i <= 0.0:
            continue
        ent -= i * math.log(i,n_classes)
    return ent

# determine which columns provide the larges information gain
def information_gain(obj,baseline):
    baseline_entropy = entropy(obj[baseline])
    ret = {}
    for column in obj:
        if column == baseline:
            continue
        if column not in ret:
            ent = entropy(obj[column])
            ret[column] = {
                'entropy': ent,                
                'gain': baseline_entropy - ent,
                'survived': numpy.nan if len(obj[column].unique()) > 10 else obj.groupby([column]).sum()[baseline] / obj.groupby([column]).count()[baseline],
                'count': numpy.nan if len(obj[column].unique()) > 10 else obj.groupby([column]).count()[baseline]
            }
    return ret
'''    
baseline_entropy = entropy(data_train['Survived'])
print 'Entropy of Survived: %f' % (baseline_entropy)
print 'Information Gain from:'
for column in data_train:    
    print '\t%s: %f, levels: %d' % (column,baseline_entropy - entropy(data_train[column]),len(scipy.stats.itemfreq(data_train[column])))

temp = data_train.groupby(['Sex','Pclass','Fare']).aggregate(numpy.sum)['Survived']
'''

# define prediction logic
# alter data into buckets

# take in the entire instance and return the prediction value
'''
def predict(obj):
    vector = obj.Gender == 0 or obj.EmbarkedFactor == 1 
    if type(obj.get('Survived')) == pandas.core.series.Series:
        return vector == obj['Survived']
    return vector


# verify logic on data_test
data_test['pass'] = predict(data_test)
evaluate = numpy.sum(data_test['pass']).astype(numpy.float)
print 'Accuracy of predictor: %f' % (evaluate / data_test.shape[0] if data_test.shape[0] > 0 else None)
'''

# open test file and generate output file
if not os.path.exists(filename_test_input):
    print 'Unable to locate file: %s, abort!' % (filename_test_input)
    exit()
data_raw = None
data_raw = pandas.read_csv(filename_test_input, quotechar='"',skipinitialspace=True)
data_raw = clean_data(data_raw)
data_raw['Survived'] = data_raw.apply(lambda x: 1 if (x.Gender == 0) | 
    (x.AgeBucket == 0) | (x.Pclass == 1)
    else 0, axis=1)
data_output = data_raw[['PassengerId','Survived']]
output_handle = csv.writer(open(filename_output,'wb'))
output_handle.writerow(['PassengerId','Survived'])
for row in data_output.iterrows():
    # idx[0] = index data
    # idx[1] = pandas.core.series.Series data
    output_handle.writerow([row[1]['PassengerId'],int(row[1]['Survived'])])

data_raw = None
data_output = None
output_handle = None
filename_test_input = None
