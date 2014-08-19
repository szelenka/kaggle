# -*- coding: utf-8 -*-

# import libaries
import os, csv, numpy, pandas
from scipy.stats import binom

# specify variables
filename_train_input = 'data/train.csv'
filename_test_input = 'data/test.csv'
filename_output = 'data/pytest2.csv'

# open csv file
if not os.path.exists(filename_train_input):
    print 'Unable to locate file: %s, abort!' % (filename_train_input)
    exit()

data_raw = pandas.read_csv(filename_train_input, delimiter='"',)

csv_file_object = csv.reader(open(filename_train_input,'rb'))

# read csv data to memory
header = csv_file_object.next()
data_raw = []
for row in csv_file_object:
    data_raw.append(row)
    
data_raw = numpy.array(data_raw)
csv_file_object = None
filename_train_input = None
row = None

# split into train/test sets
numpy.random.seed(777)
data_split = numpy.random.binomial(1,.75,data_raw.shape[0]) == 1
data_train = data_raw[data_split,]
data_test = data_raw[-data_split,]
print 'Split set is Train: %d, Test: %d' % (data_train.shape[0], data_test.shape[0])

# calculating totals
number_passengers = numpy.size(data_train[0::,1].astype(numpy.float))
number_survived = numpy.sum(data_train[0::,1].astype(numpy.float))
proportion_survivors = number_survived / number_passengers if number_passengers > 0 else None

# get a boolean vector 
women_only_stats = data_train[0::,4] == 'female'
men_only_stats = data_train[0::,4] != 'female'

# subset data on vector, extracting only the column in position '1'
women_onboard = data_train[women_only_stats,1].astype(numpy.float)
men_onboard = data_train[men_only_stats,1].astype(numpy.float)

# discover proportions
proportion_women_survived = numpy.sum(women_onboard) / numpy.size(women_onboard) if numpy.size(women_onboard) > 0 else None
proportion_men_survived = numpy.sum(men_onboard) / numpy.size(men_onboard) if numpy.size(men_onboard) > 0 else None

# print out 
print 'Proportion of survivors: women: %f, men: %f' % (proportion_women_survived, proportion_men_survived)



# define prediction logic
# take in the entire instance and return the prediction value
def predict(obj):
    return '1' if obj[3] == 'female' else '0'




# verify logic on data_test
evaluate = []
for row in data_test:
    evaluate.append(row[1] == predict(row))

print 'Accuracy of predictor: %f' % (numpy.sum(evaluate).astype(numpy.float) / data_test.shape[0] if data_test.shape[0] > 0 else None)


# open test file and generate output file
if not os.path.exists(filename_test_input):
    print 'Unable to locate file: %s, abort!' % (filename_test_input)
    exit()
csv_file_object = csv.reader(open(filename_test_input,'rb'))
output_handle = csv.writer(open(filename_output,'wb'))
header = csv_file_object.next()
output_handle.writerow(['PassengerId','Survived'])
data_raw = []
for row in csv_file_object:
    output_handle.writerow([row[0],predict(row)])
csv_file_object = None
output_handle = None
filename_test_input = None
row = None
