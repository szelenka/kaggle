# -*- coding: utf-8 -*-

# import libaries
import os, csv, numpy, pandas
from scipy.stats import binom

# specify variables
os.chdir('/Users/szelenka/Documents/_svn/datasciencecoursera/kaggle/titantic')

filename_train_input = 'data/train.csv'
filename_test_input = 'data/test.csv'
filename_output = 'data/pytest2a.csv'

# open csv file
if not os.path.exists(filename_train_input):
    print 'Unable to locate file: %s, abort!' % (filename_train_input)
    exit()
csv_file = csv.reader(open(filename_train_input))
header = csv_file.next()
data_raw = []
for row in csv_file:
    data_raw.append(row)
data_raw = numpy.array(data_raw)
#data_raw = pandas.read_csv(filename_train_input, quotechar='"',skipinitialspace=True)
csv_file = None
filename_train_input = None

# split into train/test sets
numpy.random.seed(777)
split_vector = numpy.random.binomial(1,.75,data_raw.shape[0]) == 1
data_train = data_raw[split_vector]
data_test = data_raw[-split_vector]
print 'Split set is Train: %d, Test: %d' % (data_train.shape[0], data_test.shape[0])

# calculating totals
number_passengers = data_train.shape[0]
number_survived = numpy.sum(data_train[0::,1].astype(numpy.float))
proportion_survivors = float(number_survived) / number_passengers if number_passengers > 0 else None

# get a boolean vector 
women_only_vector = data_train[0::,4] == 'female'
men_only_vector = data_train[0::,4] != 'female'

# subset data on vector, extracting only the column in position '1'
women_onboard = data_train[women_only_vector,1].astype(numpy.float)
men_onboard = data_train[men_only_vector,1].astype(numpy.float)

# discover proportions
proportion_women_survived = numpy.sum(women_onboard).astype(numpy.float) / women_onboard.shape[0] if women_onboard.shape[0] > 0 else None
proportion_men_survived = numpy.sum(men_onboard).astype(numpy.float) / men_onboard.shape[0] if men_onboard.shape[0] > 0 else None

# print out 
print 'Proportion of survivors: women: %f, men: %f' % (proportion_women_survived, proportion_men_survived)


# add ceiling to Fare
fare_ceiling = 40
data_train[ data_train[0::,9].astype(numpy.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

# unique classes in Pclass
number_of_classes = len(numpy.unique(data_train[0::,2]))

survival_table = numpy.zeros((2,number_of_classes, number_of_price_brackets))

# iterate through ranges to calculate totals for each category
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        women_only_stats = data_train[
            (data_train[0::,4] == 'female') &
            (data_train[0::,2].astype(numpy.float) == i+1) &
            (data_train[0::,9].astype(numpy.float) >= j*fare_bracket_size) &
            (data_train[0::,9].astype(numpy.float) < (j+1)*fare_bracket_size)
        , 1]
        men_only_stats = data_train[
            (data_train[0::,4] != 'female') &
            (data_train[0::,2].astype(numpy.float) == i+1) &
            (data_train[0::,9].astype(numpy.float) >= j*fare_bracket_size) &
            (data_train[0::,9].astype(numpy.float) < (j+1)*fare_bracket_size)
        , 1]

        # preserve values
        survival_table[0,i,j] = numpy.mean(women_only_stats.astype(numpy.float))
        survival_table[1,i,j] = numpy.mean(men_only_stats.astype(numpy.float))

        # take care of NaN values
        survival_table[survival_table != survival_table] = 0

# create boolean survived using 50% as cutoff
cuttoff = .5
survival_table[survival_table < cuttoff] = 0
survival_table[survival_table >= cuttoff] = 1

# open test file and generate output file
if not os.path.exists(filename_test_input):
    print 'Unable to locate file: %s, abort!' % (filename_test_input)
    exit()
csv_file = csv.reader(open(filename_test_input))
header = csv_file.next()
output_handle = csv.writer(open(filename_output,'wb'))
output_handle.writerow(['PassengerId','Survived'])
# 8 == fare
# 3 == sex
# 1 == Pclass
for row in csv_file:
    bin_fare = None
    for j in xrange(number_of_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets-1
            break
        if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size:
            bin_fare = j
            break
    if row[3]== 'female':
        output_handle.writerow([row[0], '%d' % int(survival_table[0,float(row[1])-1, bin_fare])])
    else:
        output_handle.writerow([row[0], '%d' % int(survival_table[1,float(row[1])-1, bin_fare])])

data_raw = None
data_output = None
output_handle = None
filename_test_input = None
