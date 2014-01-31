'''
#==============================================================================
multinomial_NB.py
/Users/aelshen/Documents/Dropbox/School/CLMS 2013-2014/Winter 2014/Ling 572-Advanced Statistical Methods for NLP/hw3_572_aelshen/src/multinomial_NB.py
Created on Jan 24, 2014
@author: aelshen
#==============================================================================
'''
import math
import os
import sys
import time
from collections import defaultdict
#==============================================================================
#--------------------------------Constants-------------------------------------
#==============================================================================
DEBUG = True

#==============================================================================
#-----------------------------------Main---------------------------------------
#==============================================================================
def main():
    if len(sys.argv) < 7:
        print("multinomial_NB requires two arguments:"
              + os.linesep + "\t(1)training data"
              + os.linesep + "\t(2)test data"
              + os.linesep + "\t(3)class prior delta"
              + os.linesep + "\t(4)conditional prob delta"
              + os.linesep + "\t(5)model_file"
              + os.linesep + "\t(6)sys_output")
        sys.exit()
    training_data = sys.argv[1]
    test_data = sys.argv[2]
    class_prior_delta = sys.argv[3]
    cond_prob_delta = sys.argv[4]
    model_file = sys.argv[5]
    sys_output = sys.argv[6]

    hw3 = multinomial_NB(training_data, test_data, class_prior_delta,
                         cond_prob_delta, model_file, sys_output)

#==============================================================================    
#---------------------------------Functions------------------------------------
#==============================================================================


#==============================================================================    
#----------------------------------Classes-------------------------------------
#==============================================================================
class multinomial_NB:
    def __init__(self, train_file, test_file, class_prior_delta, 
                 cond_prob_delta, model_file, sys_output):
        self.model_file = open(model_file, 'w')
        #opens the file briefly in order to empty it, then reopen it with 
        #privileges set to 'a'
        self.sys_output = open(sys_output, 'w').close()
        self.sys_output = open(sys_output, 'a')
        self.train_file = open(train_file, 'r')
        self.test_file = open(test_file, 'r')
        
        self.class_prior_delta = float(class_prior_delta)
        self.cond_prob_delta = float(cond_prob_delta)
        
        
        #a dict of classes, to be used to count the number
        #of instances per label
        self.classes = defaultdict(int)
        #2D hash to keep track of the count of a given feature 
        #for each class
        self.feature_count_per_class = defaultdict(lambda: defaultdict(int))
        #the total number of feature occurrences for a class
        self.total_features_per_class = defaultdict(int)
        #a count of instances (i.e. documents)
        self.instance_count = 0
        #the vocabulary
        self.vocabulary = defaultdict(int)
        #2D hash to keep track of the conditional probability
        #of a feature for a given class
        self.cond_logprob = defaultdict(lambda: defaultdict(int))
        #a list of the results of classifying the training/test data
        self.training_classifications = []
        self.testing_classifications = []
        #2D hashes to keep track of the number of times a document with 
        #label X was given a label of Y
        self.training_tallies = defaultdict(lambda: defaultdict(int))
        self.testing_tallies= defaultdict(lambda: defaultdict(int))
        
        self.Train()
        self.Test()
        self.PrintModel()
        self.PrintSys()
        self.PrintConfusionMatrix()

    def Train(self):
        #for each line in the training data
        for line in self.train_file.readlines():
            line = line.strip().split()
            #first index of each line is the true label of the document
            label = line[0]
            
            self.classes[label] += 1
            self.instance_count += 1
            
            #the remaining tokens in the line are feature:count pairs
            for attribute in line[1:]:
                feature,count = attribute.split(":")
                count = int(count)
                if count > 0:
                    #if this feature occurs in a document of class x
                    #increment the count of this feature given class x
                    #by the count of that feature
                    self.feature_count_per_class[label][feature] += count

                    #increment the total number of features
                    self.total_features_per_class[label] += count
                    
                    #add feature to the vocabulary
                    self.vocabulary[feature] += count
            #end for attribute in attributes[1:]
        #end for line in self.train_file:
        
        for cls in self.classes:
            for word in self.vocabulary:
                #find the conditional probability of this feature given this class
                self.cond_logprob[word][cls] = self.FindMultinomialCondProb(word, cls)
        self.train_file.seek(0)
    
    def FindMultinomialCondProb(self, word, cls):
        p_wc = float( (self.cond_prob_delta + self.feature_count_per_class[cls][word]) ) \
               / (len(self.vocabulary) * self.cond_prob_delta + self.total_features_per_class[cls])
        return math.log10(p_wc)
               
        
        
    #arguments: expects an already opened file
    def Test(self):
        #to time the length of testing
        start = time.time()
        
        #for each line in the training data
        for line in self.train_file.readlines():
            line = line.strip().split()
            #first item is the true label of the instance
            true_label = line[0]
            
            #calculate the prior probability of each class
            #and add that prior probability to the class probability 
            #for this instance
            class_prob = defaultdict(float)
            for cls in self.classes:
                class_prob[cls] = math.log10(float( self.class_prior_delta + self.classes[cls]) \
                                     /(len(self.classes)*self.class_prior_delta + self.instance_count) )

            for attribute in line[1:]:
                word, count = attribute.split(":")
                count = int(count)
                #if the feature has a non-zero count, it will be used in classification
                if count > 0:
                    for cls in self.classes:
                        #class probability of cls is incremented by the logprob of the 
                        #conditional probability of the word given class cls times the 
                        #frequency of the word
                        if word in self.feature_count_per_class[cls]:
                            class_prob[cls] += count * self.cond_logprob[word][cls]

                    #end for cls in self.classes:
            #end for attribute in line[1:]: 
            
            
            #the following calculates a common factor by which all the probabilities 
            #will be changed, so that we can sum the probabilities.
            #this common factor will be subtracted from the  
            #calculated logprob thus far.
            common_factor = float("-inf")
            for cls in class_prob:
                if class_prob[cls] > common_factor:
                    common_factor = class_prob[cls]
            #end for cls in class_prob:

            #calculate the sum of all probabilities for all classes
            #to be used as the denominator in the following probability
            denominator = 0.0
            for i in class_prob:
                denominator += math.pow(10, class_prob[i] - common_factor)

            for cls in class_prob:
                numerator = math.pow(10, class_prob[cls] - common_factor)
                class_prob[cls] = numerator/denominator

                
            #sort the dictionary of probabilities such that items in the dict
            #are in descending order of probability
            class_prob = sorted(class_prob.items(), key=lambda x:x[1], reverse=True)
            
            #append to self.classifications a tuple of the true_label for the instance,
            #and the dict containing all the classes and their probabilities, 
            #sorted by probability
            self.training_classifications.append( ( true_label, class_prob ) )
            
            #we compare there the true label with the label produced by the system
            #and add that to a 2D hash, so we can count how many time's the true label
            #was correctly or incorrectly classified
            self.training_tallies[true_label][class_prob[0][0]] += 1
        #end for line in file.readlines():



        #a repeat of the above steps, only this time utilizing the test data
        for line in self.test_file.readlines():
            line = line.strip().split()
            true_label = line[0]
            
            class_prob = defaultdict(float)
            for cls in self.classes:
                class_prob[cls] = math.log10(float( self.class_prior_delta + self.classes[cls]) \
                                     /len(self.classes)*self.class_prior_delta + self.instance_count)

            for attribute in line[1:]:
                word, count = attribute.split(":")
                count = int(count)
                if count > 0:
                    for cls in self.classes:
                        if word in self.feature_count_per_class[cls]:
                            class_prob[cls] += count * self.cond_logprob[word][cls]

                    #end for cls in self.classes:
            #end for attribute in line[1:]: 

            common_factor = float("-inf")
            for cls in class_prob:
                if class_prob[cls] > common_factor:
                    common_factor = class_prob[cls]
            #end for cls in class_prob:

            #calculate the sum of all probabilities for all classes
            #to be used as the denominator in the following probability
            denominator = 0.0
            for i in class_prob:
                denominator += math.pow(10, class_prob[i] - common_factor)

            for cls in class_prob:
                numerator = math.pow(10, class_prob[cls] - common_factor)
                class_prob[cls] = numerator/denominator

                

            #class_prob = sorted(class_prob, key=class_prob.get, reverse=True)
            class_prob = sorted(class_prob.items(), key=lambda x:x[1], reverse=True)
            self.testing_classifications.append( ( true_label, class_prob ) )
            self.testing_tallies[true_label][class_prob[0][0]] += 1
        #end for line in file.readlines():
        
        self.test_time = time.time() - start
    
    def PrintModel(self):
        self.model_file.write("%%%%% prior prob P(c) %%%%%" + os.linesep)
        #print the prob for each class, followed by its logprob
        for cls in self.classes:
            x = float( self.class_prior_delta + self.classes[cls]) \
                      /(len(self.classes)*self.class_prior_delta + self.instance_count)
            self.model_file.write(cls + "\t" 
                                  + str( x )
                                  + "\t" 
                                  + str( math.log10(x) )
                                  + os.linesep)

        
        self.model_file.write(os.linesep)
        self.model_file.write("%%%%% conditional prob P(f|c) %%%%%" + os.linesep)
        
        #for each class, print out all the features in that class, the prob of that feature
        #and the logprob
        for cls in self.classes:
            self.model_file.write("%%%%% conditional prob P(f|c) c=" + cls +" %%%%%" + os.linesep)
            
            for word in sorted(self.vocabulary):
                val = math.pow(10, self.cond_logprob[word][cls])
                self.model_file.write(word + "\t" + cls + "\t" + str( val ) + "\t")
                if val == 0:
                    self.model_file.write("-inf" + os.linesep)
                else:
                    self.model_file.write( str( self.cond_logprob[word][cls] ) + os.linesep)
        
        
            #end for feature in self.feature_count_per_class:
            self.model_file.write(os.linesep)
        #end for cls in self.models:
        
    def PrintSys(self):
        #from the list of classifications produced in the testing phase, 
        #run through each one, and print the true label ([0] of tuple)
        #and then run through the dict ([1] of tuple) to print the 
        #class probabilities calculated for that instance.
        for i in range( len(self.training_classifications) ):
            self.sys_output.write("train" + str(i) + ": " + self.training_classifications[i][0])
            for j in self.training_classifications[i][1]:
                self.sys_output.write("\t" + j[0] + "\t" + str(j[1]))    
            self.sys_output.write(os.linesep)          
            
                            
        for i in range( len(self.testing_classifications) ):
            self.sys_output.write("test" + str(i) + ": " + self.testing_classifications[i][0])
            for j in self.testing_classifications[i][1]:
                self.sys_output.write("\t" + j[0] + "\t" + str(j[1]) )     
            self.sys_output.write(os.linesep)        


    def PrintConfusionMatrix(self):
            print("Confusion matrix for the training data:")
            print("row is the truth, column is the system output" + os.linesep)

            correct_labels = 0
            total_labels = 0

            print("\t"*2, end="")
            for cls in self.classes:
                    print(cls + "\t", end="")
            print("")
            for true_class in self.classes:
                    print(true_class + "\t", end="")
                    #from the training_testing tallies produced in Test()
                    #print the number of times the true label was classified as Y
                    #for each y in self.classes
                    for nb_class in self.classes:
                            print(str(self.training_tallies[true_class][nb_class]) + "\t", end="")
                            total_labels += self.training_tallies[true_class][nb_class]
                            if true_class == nb_class:
                                    correct_labels += self.training_tallies[true_class][nb_class]
                    print("")

            print(os.linesep + "Training accuracy="+str(float(correct_labels)/total_labels) + os.linesep)

            print(os.linesep*2)
            print("Confusion matrix for the test data:")
            print("row is the truth, column is the system output" + os.linesep)

            correct_labels = 0
            total_labels = 0

            print("\t"*2, end="")
            for cls in self.classes:
                    print(cls + "\t", end="")
            print("")
            for true_class in self.classes:
                    print(true_class + "\t", end="")
                    #from the training_testing tallies produced in Test()
                    #print the number of times the true label was classified as Y
                    #for each y in self.classes
                    for nb_class in self.classes:
                            print(str(self.testing_tallies[true_class][nb_class]) + "\t", end="")
                            total_labels += self.testing_tallies[true_class][nb_class]
                            if true_class == nb_class:
                                    correct_labels += self.testing_tallies[true_class][nb_class]
                    print("")

            print(os.linesep + " Test accuracy="+str(float(correct_labels)/total_labels) + os.linesep)
            print("Test time: " + str(self.test_time))
    #close all files on destruct
    def __exit__(self, type, value, traceback):
        self.train_file.close()
        self.test_file.close()
        self.model_file.close()
        self.sys_output.close()
        
if __name__ == "__main__":
    sys.exit( main() )
