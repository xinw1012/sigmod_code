import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import classification as lg
import digit_class as digits
#import digits_function as digitsfunc
from datetime import datetime
import time
from sklearn import svm, ensemble


###### Load Data #################

train_x, train_y = digits.load_digits("/Users/xinw/Documents/projects/velox-centipede/data", digits_filename = "mnist_train.csv")
Z = digits.normalize_digits(train_x)
test_x, test_y = digits.load_digits("/Users/xinw/Documents/projects/velox-centipede/data", digits_filename = "mnist_test.csv")
test_Z = digits.normalize_digits(test_x)
#Z = digits.fourier_project(train_x)



### Partial Concept Drift 

####### Initialization ##########
tasks = digits.create_mtl_datasets(Z, train_y, nTasks=100, taskSize=20, testSize=40)
#xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
xs = []
ys = []
dy = []
ss = []
k = 20
k_svm = 10
train_iter = 2

"""
## Strategy: Train All data
oracle_train_all_errors = []
oracle_train_all = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
svm_train_all = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
test_xs,test_ys,test_ss = digits.generate_additional_data (tasks, oracle_train_all, svm_train_all,3)
for j in range(train_iter):
    oracle_train_all.train_all_fs()
    oracle_train_all.train_ws()
oracle_err = lg.seg_model_error_01(oracle_train_all, test_xs, test_ys, test_ss, num=20)
oracle_train_all_errors.append(oracle_err)
#oracle_mtl_errors.append(oracle_err)
print "Initial Error OracleMTL---Train All: %f" % oracle_err

svm_train_all.train_all_fs()
svm_train_all.train_ws()
svm_err = lg.seg_model_error_01(svm_train_all,test_xs,test_ys,test_ss)
print 'Initial Error SVM_mtl---Train All: ', svm_err
svm_train_all_errors = [svm_err]

"""
## strategy: retrain-new
oracle_retrain_new_errors = []
oracle_retrain_new = lg.LgSegmentModel(xs,ys,ss,'train-all',k)
svm_retrain_new = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
test_xs,test_ys,test_ss = digits.generate_additional_data (tasks, oracle_retrain_new, svm_retrain_new,3)
for j in range(train_iter):
    oracle_retrain_new.train_all_fs()
    oracle_retrain_new.train_ws()
oracle_retrain_new.strategy = 'retrain-new'    

svm_retrain_new.train_all_fs()
svm_retrain_new.train_ws()
svm_retrain_new_errors = []
svm_retrain_new.strategy = 'retrain-new'
oracle_err = lg.seg_model_error_01(oracle_retrain_new, test_xs, test_ys, test_ss, num=20)
oracle_retrain_new_errors.append(oracle_err)
svm_err = lg.seg_model_error_01(svm_retrain_new,test_xs,test_ys,test_ss)
svm_retrain_new_errors.append(svm_err)
print "Initial Error OracleMTL---Train All: %f" % oracle_err
print 'Initial Error SVM_mtl---Train All: ', svm_err


"""
## strategy: average-weight
oracle_average_weight_errors = []
oracle_average_weight = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
svm_average_weight = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
test_xs,test_ys,test_ss = digits.generate_additional_data (tasks, oracle_average_weight, svm_average_weight,3)
for j in range(train_iter):
    oracle_average_weight.train_all_fs()
    oracle_average_weight.train_ws()
oracle_average_weight.strategy = 'average-weight'    

svm_average_weight.train_all_fs()
svm_average_weight.train_ws()
svm_average_weight_errors = []
svm_average_weight.strategy = 'average-weight'

oracle_err = lg.seg_model_error_01(oracle_average_weight, test_xs, test_ys, test_ss, num=20)
oracle_retrain_new_errors.append(oracle_err)
svm_err = lg.seg_model_error_01(svm_average_weight,test_xs,test_ys,test_ss)
svm_retrain_new_errors.append(svm_err)
print "Initial Error OracleMTL---Train All: %f" % oracle_err
print 'Initial Error SVM_mtl---Train All: ', svm_err

## strategy: last-point
oracle_last_point_errors = []
oracle_last_point = lg.LgSegmentModel(xs,ys,ss,'train-all',k)
svm_last_point = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
test_xs,test_ys,test_ss = digits.generate_additional_data (tasks, oracle_last_point, svm_last_point,3)
for j in range(train_iter):
    oracle_last_point.train_all_fs()
    oracle_last_point.train_ws()
oracle_last_point.strategy = 'last-point'    

svm_last_point.train_all_fs()
svm_last_point.train_ws()
svm_last_point_errors = []
svm_last_point.strategy = 'last-point'

oracle_err = lg.seg_model_error_01(oracle_last_point, test_xs, test_ys, test_ss, num=20)
oracle_retrain_new_errors.append(oracle_err)
svm_err = lg.seg_model_error_01(svm_last_point,test_xs,test_ys,test_ss)
svm_retrain_new_errors.append(svm_err)
print "Initial Error OracleMTL---Train All: %f" % oracle_err
print 'Initial Error SVM_mtl---Train All: ', svm_err


## strategy: Gradient-step
oracle_gradient_step_errors = []
oracle_gradient_step = lg.LgSegmentModel(xs,ys,ss,'train-all',k)
svm_gradient_step = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
test_xs,test_ys,test_ss = digits.generate_additional_data (tasks, oracle_gradient_step, svm_gradient_step,3)

for j in range(train_iter):
    oracle_gradient_step.train_all_fs()
    oracle_gradient_step.train_ws()
oracle_gradient_step.strategy = 'Gradient-step'    


svm_gradient_step.train_all_fs()
svm_gradient_step.train_ws()
svm_gradient_step_errors = []
svm_gradient_step.strategy = 'Gradient-step'

oracle_err = lg.seg_model_error_01(oracle_gradient_step, test_xs, test_ys, test_ss, num=20)
oracle_retrain_new_errors.append(oracle_err)
svm_err = lg.seg_model_error_01(svm_gradient_step,test_xs,test_ys,test_ss)
svm_retrain_new_errors.append(svm_err)
print "Initial Error OracleMTL---Train All: %f" % oracle_err
print 'Initial Error SVM_mtl---Train All: ', svm_err

"""
print '\n\n Finish Initialization!'


####### Update Strategy Experiment #################
# number of points range from 20-200, step = 10
iters = [i+21 for i in range(20)]
for i in iters:
    print 'update-strategy: # of points: ', i
    #if i%20 == 1:
    if i == 21:
        print 'partial concept drift!'
        print
        #test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_train_all,svm_train_all,i,'partial')
        
       
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_retrain_new,svm_retrain_new,i,'partial')
        """
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_average_weight,svm_average_weight,i,'partial')
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_last_point,svm_last_point,i,'partial')
    
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_gradient_step,svm_gradient_step,i,'partial')
        """
    else:
        #test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_train_all,svm_train_all,i,'same')
        
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_retrain_new,svm_retrain_new,i,'same')
        """
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_average_weight,svm_average_weight,i,'same')
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_last_point,svm_last_point,i,'same')
    
        test_xs,test_ys,test_ts = digits.generate_data_concept_drift(Z, train_y, oracle_gradient_step,svm_gradient_step,i,'same')
        """
    ## oracle
    """
    oracle_train_all.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_train_all, test_xs, test_ys, test_ts, num=20)
    oracle_train_all_errors.append(oracle_err)
    print 'Testing Error of oracle model-- train-all is ', oracle_err
    """
    oracle_retrain_new.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_retrain_new, test_xs, test_ys, test_ts, num=20)
    oracle_retrain_new_errors.append(oracle_err)
    print 'Testing Error of oracle model-- retrain_new is ', oracle_err
    """
    oracle_average_weight.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_average_weight, test_xs, test_ys, test_ts, num=20)
    oracle_average_weight_errors.append(oracle_err)
    print 'Testing Error of oracle model-- average_weight is ', oracle_err
    
    oracle_last_point.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_last_point, test_xs, test_ys, test_ts, num=20)
    oracle_last_point_errors.append(oracle_err)
    print 'Testing Error of oracle model-- last_point is ', oracle_err
    
    oracle_gradient_step.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_gradient_step, test_xs, test_ys, test_ts, num=20)
    oracle_gradient_step_errors.append(oracle_err)
    print 'Testing Error of oracle model-- gradient_step is ', oracle_err    
    """
    ### svm
    """
    svm_train_all.train_ws()
    svm_err = lg.seg_model_error_01(svm_train_all, test_xs, test_ys, test_ts)
    svm_train_all_errors.append(svm_err)
    print 'Testing Error of svm model-- train-all is ', svm_err
    
    """
    svm_retrain_new.train_ws()
    svm_err = lg.seg_model_error_01(svm_retrain_new, test_xs, test_ys, test_ts)
    svm_retrain_new_errors.append(svm_err)
    print 'Testing Error of svm model-- retrain_new is ', svm_err
    
    """
    svm_average_weight.train_ws()
    svm_err = lg.seg_model_error_01(svm_average_weight, test_xs, test_ys, test_ts)
    svm_average_weight_errors.append(svm_err)
    print 'Testing Error of svm model-- average_weight is ', svm_err
    
    svm_last_point.train_ws()
    svm_err = lg.seg_model_error_01(svm_last_point, test_xs, test_ys, test_ts)
    svm_last_point_errors.append(svm_err)
    print 'Testing Error of svm model-- last_point is ', svm_err
    
    
    svm_gradient_step.train_ws()
    svm_err = lg.seg_model_error_01(svm_gradient_step, test_xs, test_ys, test_ts)
    svm_gradient_step_errors.append(svm_err)
    print 'Testing Error of svm model-- gradient_step is ', svm_err  

    """
iters = [20]+iters
## Write Files ###

f = open('update-strategy-experiment-partial-concept-drift.txt','a')
"""
f.write('strategy: train-all\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_train_all_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_train_all_errors:
    f.write('\t'+str(t))
f.write('\n')
"""
f.write('strategy: retrain_new\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_retrain_new_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_retrain_new_errors:
    f.write('\t'+str(t))
f.write('\n')

"""
f.write('strategy: average_weight\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_average_weight_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_average_weight_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('strategy: last_point\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_last_point_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_last_point_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('strategy: gradient_step\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_gradient_step_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_gradient_step_errors:
    f.write('\t'+str(t))
f.write('\n')
"""
f.close()

## plot ###
fig,ax = plt.subplots()
#ax.plot(iters, oracle_train_all_errors, label='train_all')
ax.plot(iters, oracle_retrain_new_errors, label = 'retrain_new')
#ax.plot(iters, oracle_average_weight_errors, label = 'average_weight')
#ax.plot(iters, oracle_last_point_errors, label = 'last_point')
#ax.plot(iters, oracle_gradient_step_errors, label = 'gradient_step')
ax.set_xlabel('Size of the task')
ax.set_ylabel('Error rate')
ax.set_title('Update-strategy -- Partial-Concept-Drift --Oracle ')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()

fig,ax = plt.subplots()
#ax.plot(iters, svm_train_all_errors, label='train_all')
ax.plot(iters, svm_retrain_new_errors, label = 'retrain_new')
#ax.plot(iters, svm_average_weight_errors, label = 'average_weight')
#ax.plot(iters, svm_last_point_errors, label = 'last_point')
#ax.plot(iters, svm_gradient_step_errors, label = 'gradient_step')
ax.set_xlabel('Size of the task')
ax.set_ylabel('Error rate')
ax.set_title('Update-strategy -- Partial-Concept-Drift --svm ')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()




"""
#### user statinery ####

####### Initialization ##########
tasks = digits.create_mtl_datasets(Z, train_y, nTasks=50, taskSize=20, testSize=30)
xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
k = 20
k_svm = 10
train_iter = 6


## Strategy: Train All data
oracle_train_all_errors = []
oracle_train_all = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
for j in range(train_iter):
    oracle_train_all.train_all_fs()
    oracle_train_all.train_ws()
oracle_err = lg.seg_model_error_01(oracle_train_all, test_xs, test_ys, test_ts, num=20)
#oracle_mtl_errors.append(oracle_err)
print "Initial Error OracleMTL---Train All: %f" % oracle_err

svm_train_all = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
svm_train_all.train_all_fs()
svm_train_all.train_ws()
svm_err = lg.seg_model_error_01(svm_train_all,test_xs,test_ys,test_ts)
print 'Initial Error SVM_mtl---Train All: ', svm_err
svm_train_all_errors = []

## strategy: retrain-new
oracle_retrain_new_errors = []
oracle_retrain_new = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
for j in range(train_iter):
    oracle_retrain_new.train_all_fs()
    oracle_retrain_new.train_ws()
oracle_retrain_new.strategy = 'retrain-new'    

svm_retrain_new = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
svm_retrain_new.train_all_fs()
svm_retrain_new.train_ws()
svm_retrain_new_errors = []
svm_retrain_new.strategy = 'retrain-new'

## strategy: average-weight
oracle_average_weight_errors = []
oracle_average_weight = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
for j in range(train_iter):
    oracle_average_weight.train_all_fs()
    oracle_average_weight.train_ws()
oracle_average_weight.strategy = 'average-weight'    

svm_average_weight = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
svm_average_weight.train_all_fs()
svm_average_weight.train_ws()
svm_average_weight_errors = []
svm_average_weight.strategy = 'average-weight'

## strategy: last-point
oracle_last_point_errors = []
oracle_last_point = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
for j in range(train_iter):
    oracle_last_point.train_all_fs()
    oracle_last_point.train_ws()
oracle_last_point.strategy = 'last-point'    

svm_last_point = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
svm_last_point.train_all_fs()
svm_last_point.train_ws()
svm_last_point_errors = []
svm_last_point.strategy = 'last-point'


## strategy: Gradient-step
oracle_gradient_step_errors = []
oracle_gradient_step = lg.LgSegmentModel(xs,ys,ss,'train-all',k) 
for j in range(train_iter):
    oracle_gradient_step.train_all_fs()
    oracle_gradient_step.train_ws()
oracle_gradient_step.strategy = 'Gradient-step'    


svm_gradient_step = lg.UserDefineModel(xs,ys,dy,ss,'train-all') # using the default value 
svm_gradient_step.train_all_fs()
svm_gradient_step.train_ws()
svm_gradient_step_errors = []
svm_gradient_step.strategy = 'Gradient-step'

print '\n\n Finish Initialization!'

####### Update Strategy Experiment #################
# number of points range from 20-200, step = 10
iters = [i+20 for i in range(30)]
for i in iters:
    print 'update-strategy: # of points: ', i
    task = digits.create_mtl_datasets(Z,train_y,nTasks=50,taskSize=1,testSize=1000)
    
    test_xs,test_ys,test_ts = digits.generate_additional_data(task,oracle_train_all, svm_train_all,1)       
    test_xss,test_yss,test_tss = digits.generate_additional_data(task,oracle_retrain_new, svm_retrain_new,1)
    test_xss,test_yss,test_tss = digits.generate_additional_data(task,oracle_average_weight, svm_average_weight,1)
    test_xss,test_yss,test_tss = digits.generate_additional_data(task,oracle_last_point, svm_last_point,1)
    
    test_xss,test_yss,test_tss = digits.generate_additional_data(task,oracle_gradient_step, svm_gradient_step,1)
        
    ## oracle
    
    oracle_train_all.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_train_all, test_xs, test_ys, test_ts, num=20)
    oracle_train_all_errors.append(oracle_err)
    print 'Testing Error of oracle model-- train-all is ', oracle_err
    
    oracle_retrain_new.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_retrain_new, test_xs, test_ys, test_ts, num=20)
    oracle_retrain_new_errors.append(oracle_err)
    print 'Testing Error of oracle model-- retrain_new is ', oracle_err
    
    oracle_average_weight.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_average_weight, test_xs, test_ys, test_ts, num=20)
    oracle_average_weight_errors.append(oracle_err)
    print 'Testing Error of oracle model-- average_weight is ', oracle_err
    
    oracle_last_point.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_last_point, test_xs, test_ys, test_ts, num=20)
    oracle_last_point_errors.append(oracle_err)
    print 'Testing Error of oracle model-- last_point is ', oracle_err
    
    oracle_gradient_step.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_gradient_step, test_xs, test_ys, test_ts, num=20)
    oracle_gradient_step_errors.append(oracle_err)
    print 'Testing Error of oracle model-- gradient_step is ', oracle_err    
       
    ### svm
    
    svm_train_all.train_ws()
    svm_err = lg.seg_model_error_01(svm_train_all, test_xs, test_ys, test_ts)
    svm_train_all_errors.append(svm_err)
    print 'Testing Error of svm model-- train-all is ', svm_err
    
    svm_retrain_new.train_ws()
    svm_err = lg.seg_model_error_01(svm_retrain_new, test_xs, test_ys, test_ts)
    svm_retrain_new_errors.append(svm_err)
    print 'Testing Error of svm model-- retrain_new is ', svm_err
    
    svm_average_weight.train_ws()
    svm_err = lg.seg_model_error_01(svm_average_weight, test_xs, test_ys, test_ts)
    svm_average_weight_errors.append(svm_err)
    print 'Testing Error of svm model-- average_weight is ', svm_err
    
    svm_last_point.train_ws()
    svm_err = lg.seg_model_error_01(svm_last_point, test_xs, test_ys, test_ts)
    svm_last_point_errors.append(svm_err)
    print 'Testing Error of svm model-- last_point is ', svm_err
    
    
    svm_gradient_step.train_ws()
    svm_err = lg.seg_model_error_01(svm_gradient_step, test_xs, test_ys, test_ts)
    svm_gradient_step_errors.append(svm_err)
    print 'Testing Error of svm model-- gradient_step is ', svm_err  

    
## Write Files ###
f = open('update-strategy-experiment-user-stationery.txt','w')
f.write('strategy: train-all\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_train_all_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_train_all_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('strategy: retrain_new\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_retrain_new_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_retrain_new_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('strategy: average_weight\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_average_weight_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_average_weight_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('strategy: last_point\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_last_point_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_last_point_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('strategy: gradient_step\n')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')     

for t in oracle_gradient_step_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n\t')  
for t in svm_gradient_step_errors:
    f.write('\t'+str(t))
f.write('\n')

f.close()

## plot ###
fig,ax = plt.subplots()
ax.plot(iters, oracle_train_all_errors, label='train_all')
ax.plot(iters, oracle_retrain_new_errors, label = 'retrain_new')
ax.plot(iters, oracle_average_weight_errors, label = 'average_weight')
ax.plot(iters, oracle_last_point_errors, label = 'last_point')
ax.plot(iters, oracle_gradient_step_errors, label = 'gradient_step')
ax.set_xlabel('Size of the task')
ax.set_ylabel('Error rate')
ax.set_title('Update-strategy -- User Distribution Stationery--Oracle ')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()

fig,ax = plt.subplots()
ax.plot(iters, svm_train_all_errors, label='train_all')
ax.plot(iters, svm_retrain_new_errors, label = 'retrain_new')
ax.plot(iters, svm_average_weight_errors, label = 'average_weight')
ax.plot(iters, svm_last_point_errors, label = 'last_point')
ax.plot(iters, svm_gradient_step_errors, label = 'gradient_step')
ax.set_xlabel('Size of the task')
ax.set_ylabel('Error rate')
ax.set_title('Update-strategy -- User Distribution Stationery--svm ')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()

"""

"""

####### Add Tasks Experiment #####
k = 10
iters = range(100,1000,100)
oracle_mtl_errors = []
predefine_mtl_errors = []
sep_model_errors = []
for i in iters:
    print 'number of tasks is ', i
    tasks = digits.create_mtl_datasets(Z, train_y, nTasks=i, taskSize=10, testSize=20)
    xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
    
    ### oracle_mtl ###
    oracle_mtl = lg.LgSegmentModel(xs,ys,ss,k)      
    for j in range(6):
        oracle_mtl.train_all_fs()
        oracle_mtl.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_mtl, test_xs, test_ys, test_ts)
    oracle_mtl_errors.append(oracle_err)
    print "Done training OracleMTL: %f" % oracle_err
    
    ### predefine_mtl ###
    predefine_mtl = lg.UserDefineModel(xs,ys,dy,ss,k)
    predefine_mtl.train_all_fs()
    predefine_mtl.train_ws()
    predefine_err = lg.seg_model_error_01(predefine_mtl,test_xs,test_ys,test_ts)
    predefine_mtl_errors.append(predefine_err)
    print "Done training predefineMTL: %f" % predefine_err
    
    ### Sep models ###
    sep_models = lg.LgSegmentModel(xs,ys,ss,k).segments
    for sid, s in sep_models.iteritems():
        m = lm.Ridge()
        m.fit(s.xs, s.ys)
        s.model = m
    sep_model_test_error = lg.separate_model_error_01(sep_models, test_xs, test_ys, test_ts)
    sep_model_errors.append(sep_model_test_error)
    print "Done training Separate Models: %f" % sep_model_test_error
    
### plot ####
fig, ax = plt.subplots()
#ax.plot(iters, oracle_mtl_errors, label="oracle")
#ax.plot(iters, sep_model_errors, label="sep")
ax.plot(iters, predefine_mtl_errors,label="predefine")
ax.set_title("Error vs Number of Tasks")
ax.set_xlabel("Total Number of Tasks")
ax.set_ylabel("01 Errors")
#ax.set_yscale('log')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
#ax.set_ylim((0.0, 0.45))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()
"""

"""
######## Varying number of features ###
k_oracle = 20
k_svm = 10
nTasks=100
taskSize=100
tasks = digits.create_mtl_datasets(Z, train_y, nTasks, taskSize, testSize=20)
xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
oracle_feature_errors = []

iter_oracle = range(20,k_oracle+1)
iter_svm = range(0,k_svm*3+1,5)

for i in iter_oracle:
    print 'number of feature computed is ', i
    
    ### oracle_mtl ###
    oracle_mtl = lg.LgSegmentModel(xs,ys,ss,k_oracle)      
    for j in range(8):
        oracle_mtl.train_all_fs()
        oracle_mtl.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_mtl, test_xs, test_ys, test_ts,i)
    oracle_feature_errors.append(oracle_err)
    print "Done training OracleMTL: %f" % oracle_err

points = nTasks*taskSize

svm_mtl = lg.UserDefineModel(xs,ys,dy,ss,'svm','logistic',points,k_svm)
svm_mtl.train_all_fs()
svm_mtl.train_ws()
svm_feature_errors = []


nb_mtl = lg.UserDefineModel(xs,ys,dy,ss,'naive_bayes',k_svm)
nb_mtl.train_all_fs()
nb_mtl.train_ws()
nb_feature_errors = []

log_mtl = lg.UserDefineModel(xs,ys,dy,ss,'logistic',k_svm)
log_mtl.train_all_fs()
log_mtl.train_ws()
log_feature_errors = []

dt_mtl = lg.UserDefineModel(xs,ys,dy,ss,'decision_tree',k_svm)
dt_mtl.train_all_fs()
dt_mtl.train_ws()
dt_feature_errors = []


gb_mtl = lg.UserDefineModel(xs,ys,dy,ss,'gradient_boosting',k_svm)
gb_mtl.train_all_fs()
gb_mtl.train_ws()
gb_feature_errors = []

for i in iter_svm:
    ### predefine_mtl ###
    print 'number of features is ', i
    svm_err = lg.seg_model_error_01(svm_mtl,test_xs,test_ys,test_ts,i)
    svm_feature_errors.append(svm_err)
    print "Done training svm_MTL: %f" % svm_err
    
    nb_err = lg.seg_model_error_01(nb_mtl,test_xs,test_ys,test_ts,i)
    nb_feature_errors.append(nb_err)
    print "Done training nb_MTL: %f" % nb_err
    
    log_err = lg.seg_model_error_01(log_mtl,test_xs,test_ys,test_ts,i)
    log_feature_errors.append(log_err)
    print "Done training log_MTL: %f" % log_err
    
    dt_err = lg.seg_model_error_01(dt_mtl,test_xs,test_ys,test_ts,i)
    dt_feature_errors.append(dt_err)
    print "Done training dt_MTL: %f" % dt_err
    

    gb_err = lg.seg_model_error_01(gb_mtl,test_xs,test_ys,test_ts,i)
    gb_feature_errors.append(gb_err)
    print "Done training svm_MTL: %f" % gb_err

### plot #####

fig, ax = plt.subplots()
ax.plot(iter_oracle, oracle_feature_errors, label="oracle")
#ax.plot(iters, sep_model_errors, label="sep")
#ax.plot(feature_iter, predefine_feature_errors,label="predefine")
ax.set_title("Error vs Number of Features Used in Online Updates")
ax.set_xlabel("Number of Features")
ax.set_ylabel("01 Errors")
#ax.set_yscale('log')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


fig, ax = plt.subplots()
#ax.plot(iter_oracle, oracle_feature_errors, label="oracle")
#ax.plot(iters, sep_model_errors, label="sep")
ax.plot(iter_svm, svm_feature_errors,label=svm_mtl.f_type)
#ax.plot(iter_svm, log_feature_errors,label=log_mtl.f_type)
#ax.plot(iter_svm, dt_feature_errors,label=dt_mtl.f_type)
#ax.plot(iter_svm, gb_feature_errors,label=gb_mtl.f_type)
#ax.plot(iter_svm, nb_feature_errors,label=nb_mtl.f_type)
ax.set_title("Error vs Number of Features Used in Online Updates")
ax.set_xlabel("Number of Features")
ax.set_ylabel("01 Errors")
#ax.set_yscale('log')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()
"""