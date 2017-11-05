import numpy as np
from perceptron import Perceptron

# read csv input
"""
remember that the input file should be the original 
file minus the validation set
"""
input_data_raw = np.genfromtxt('input data file', delimiter=',') # read input data
output_data_raw = np.genfromtxt('output label file', delimiter=',') # read output data


"""
create a cross validation data section for monitoring training accuracy

1. the cross validation data section should include 20% of the original data
   which is around 6 samples in this case.

2. create a data file and the label file seperately.
"""

validation_data_input = np.genfromtxt('input data file', delimiter=',')

validation_data_output = np.genfromtxt('input data file', delimiter=',')

# perceptron initialization and fitting

pn = Perceptron(0.15, 40) #(alpha, no_of_iteration)

fit, weight = pn.fit(input_data_raw, output_data_raw)


print "final weight"
for i in range(1, 9):      
   print 'w%d = %.2f' %(i, weight[i])
print "\n"

print "input string"
print input_data_raw
print "\n"

print "final wx sum"
print pn.net_input(input_data_raw)
print "\n"

print "no. of error term in all iterations"
print pn.errors
print "\n"

# check training accuracy

count = 0 # count no of samples in the validation set
err = 0
for element_in, element_out in zip(validation_file_input, validation_file_output):
    count += 1
    predict = pn.predict(element_in)
    print predict 
    print "\n"

    if(predict != element_out):
        err += 1

print "accuracy of testing = %d\n" % (count - err) / count
