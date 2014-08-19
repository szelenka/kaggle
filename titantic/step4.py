# -*- coding: utf-8 -*-

# import libaries
import os, csv, numpy, scipy, pandas, pylab, math
from sklearn.ensemble import RandomForestClassifier

# specify variables
filename_train_input = 'data/train.csv'
filename_test_input = 'data/test.csv'
filename_output = 'data/pytest4.csv'

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
    obj['PassengerId'] = obj.PassengerId.astype(float)
    # Ticket doesn't seem relevant?
    # Pclass is factored nicely already
    obj['Pclass'] = obj.Pclass.map(lambda x: -1.0 if numpy.isnan(x) else float(x))

    # Fare should be in ranges
    obj['FareBucket'] = obj.Fare.map(lambda x: -1.0 if numpy.isnan(x) or x < 0 else 0.0 if x < 10 else 1.0 if x < 20 else 2.0 if x < 30 else 3.0)
    
    # Name should be LastName only (to indicate family status)
    obj['LastName'] =  obj['Name'].str.replace(r'^(.*?),.*$','\\1')
    
    # Sex to Gender where female = 0 and male = 1
    obj['Gender'] = obj.Sex.map(lambda x: 0.0 if x.lower() == 'female' else 1.0 if x.lower() == 'male' else -1.0)
    
    # Age impute missing values then to Factor
    obj['AgeBucket'] = obj.Age.map(lambda x: -1 if numpy.isnan(x) else int(round(x/20)))
    age_medians = obj[['Gender','Pclass','FareBucket','AgeBucket']].groupby(['Gender','Pclass','FareBucket']).median()
    # attempt to impute missing values round to whole integer
    obj['AgeBucket'] = obj.apply(lambda x: float(x.AgeBucket) if x.AgeBucket != -1 and not numpy.isnan(x.AgeBucket) else float(age_medians['AgeBucket'][x.Gender][x.Pclass][x.FareBucket]), axis=1)
    # second pass, with less groping
    age_medians = obj[['Gender','Pclass','AgeBucket']].groupby(['Gender','Pclass']).median()
    obj['AgeBucket'] = obj.apply(lambda x: float(x.AgeBucket) if x.AgeBucket != -1 and not numpy.isnan(x.AgeBucket) else float(age_medians['AgeBucket'][x.Gender][x.Pclass]), axis=1)
    # third pass
    age_medians = obj[['Gender','AgeBucket']].groupby(['Gender']).median()
    obj['AgeBucket'] = obj.apply(lambda x: float(x.AgeBucket) if x.AgeBucket != -1 and not numpy.isnan(x.AgeBucket) else float(age_medians['AgeBucket'][x.Gender]), axis=1)

    # SibSp and Parch into a single FamilySize
    obj['FamilySize'] = obj['SibSp'] + obj['Parch']
    obj['FamilySize'] = obj.FamilySize.map(lambda x: float(x) if x < 3 else 4.0)
    
    # Cabin should be just the alphabetic character
    obj['CabinLevel'] = obj['Cabin'].str.replace(r'^([a-zA-Z]).*$','\\1')
    obj['CabinLevel'] = obj.CabinLevel.map({numpy.nan:-1,'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7}).astype(int)
    cabin_medians = obj[['Gender','FareBucket','Pclass','CabinLevel']].groupby(['Gender','Pclass','FareBucket']).median()
    obj['CabinLevel'] = obj.apply(lambda x: float(x.CabinLevel) if x.CabinLevel != -1 and not numpy.isnan(x.CabinLevel) else float(cabin_medians['CabinLevel'][x.Gender][x.Pclass][x.FareBucket]), axis=1)
    
    # Embark impute missing values then to Factor
    obj['EmbarkedFactor'] = obj.Embarked.map({numpy.nan:-1.0,'S':0.0,'C':1.0,'Q':2.0}).astype(float)
    
    # remove unused columns
    return obj.drop(['Name','Age','Sex','Ticket','Cabin','Embarked','SibSp','Parch','Fare','LastName'],axis=1)    
    

# split into train/test sets
numpy.random.seed(777)
split_vector = numpy.random.binomial(1,.75,data_raw.shape[0]) == 1
data_clean = clean_data(data_raw)
data_train = data_clean[split_vector]
data_test = data_clean[-split_vector]
print 'Split set is Train: %d, Test: %d' % (data_train.shape[0], data_test.shape[0])

def get_forest(obj,columns):
    labels = obj['Survived'].astype(float).values
    features = obj[list(columns)].values
    forest = RandomForestClassifier(n_estimators=100)
    return forest.fit(features,labels)

columns = ['Pclass','FareBucket','Gender','AgeBucket','FamilySize','CabinLevel','EmbarkedFactor']
forest = get_forest(data_train,columns)
data_test['Prediction'] = forest.predict(data_test[list(columns)].values)
accuracy = sum(data_test.Survived == data_test.Prediction).astype(float) / data_test.shape[0]

print "Accuracty of predictor: %2f" % (accuracy) 

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


# open test file and generate output file
if not os.path.exists(filename_test_input):
    print 'Unable to locate file: %s, abort!' % (filename_test_input)
    exit()

forest = get_forest(data_clean,columns)

data_raw = None
data_raw = pandas.read_csv(filename_test_input, quotechar='"',skipinitialspace=True)
data_raw = clean_data(data_raw)
data_raw['Survived'] = forest.predict(data_raw[list(columns)].values)
data_output = data_raw[['PassengerId','Survived']]
output_handle = csv.writer(open(filename_output,'wb'))
output_handle.writerow(['PassengerId','Survived'])
for row in data_output.iterrows():
    # idx[0] = index data
    # idx[1] = pandas.core.series.Series data
    output_handle.writerow([int(row[1]['PassengerId']),int(row[1]['Survived'])])

data_raw = None
data_output = None
output_handle = None
filename_test_input = None
