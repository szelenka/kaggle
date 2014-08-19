# -*- coding: utf-8 -*-

# import libaries
import os, csv, numpy, pandas
from scipy.stats import binom

# specify variables
os.chdir('/Users/szelenka/Documents/_svn/datasciencecoursera/kaggle/titantic')

filename_train_input = 'data/train.csv'
filename_test_input = 'data/test.csv'
filename_output = 'data/pytest2.csv'

# open csv file
if not os.path.exists(filename_train_input):
    print 'Unable to locate file: %s, abort!' % (filename_train_input)
    exit()

data_raw = pandas.read_csv(filename_train_input, quotechar='"',skipinitialspace=True)
filename_train_input = None

# split into train/test sets
numpy.random.seed(777)
split_vector = numpy.random.binomial(1,.75,data_raw.shape[0]) == 1
data_train = data_raw[split_vector]
data_test = data_raw[-split_vector]
print 'Split set is Train: %d, Test: %d' % (data_train.shape[0], data_test.shape[0])

# calculating totals
number_passengers = data_train.shape[0]
number_survived = numpy.sum(data_train['Survived'])
proportion_survivors = float(number_survived) / number_passengers if number_passengers > 0 else None

# get a boolean vector 
women_only_vector = data_train['Sex'] == 'female'
men_only_vector = data_train['Sex'] != 'female'

# subset data on vector, extracting only the column in position '1'
women_onboard = data_train.loc[women_only_vector,'Survived']
men_onboard = data_train.loc[men_only_vector,'Survived']

# discover proportions
proportion_women_survived = numpy.sum(women_onboard).astype(numpy.float) / women_onboard.shape[0] if women_onboard.shape[0] > 0 else None
proportion_men_survived = numpy.sum(men_onboard).astype(numpy.float) / men_onboard.shape[0] if men_onboard.shape[0] > 0 else None

# print out 
print 'Proportion of survivors: women: %f, men: %f' % (proportion_women_survived, proportion_men_survived)



# define prediction logic
# take in the entire instance and return the prediction value
def predict(obj):
    vector = obj['Sex'] == 'female'
    if type(obj.get('Survived')) == pandas.core.series.Series:
        return vector == obj['Survived']
    return vector


# verify logic on data_test
data_test['pass'] = predict(data_test)
evaluate = numpy.sum(data_test['pass']).astype(numpy.float)
print 'Accuracy of predictor: %f' % (evaluate / data_test.shape[0] if data_test.shape[0] > 0 else None)


# open test file and generate output file
if not os.path.exists(filename_test_input):
    print 'Unable to locate file: %s, abort!' % (filename_test_input)
    exit()
data_raw = None
data_raw = pandas.read_csv(filename_test_input, quotechar='"',skipinitialspace=True)
data_raw['Survived'] = predict(data_raw)
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
