# CODING FOR MNSIT DATA SET CLASSIFICATION 
import numpy as np
import pandas as pd
import keras
from numpy import argmax

from keras.utils import to_categorical


np.set_printoptions(threshold=np.inf)
def normalize(df):
 
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()                                                           #  this is applicable in pandas data frame where  different  samples in different rows and all the values for different samples  of a particular feature in a column 
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value)

    """
    
    df_max_scaled = df.copy()
  
# apply normalization techniques
    for column in df_max_scaled.columns:

        if  df_max_scaled[column].abs().max()!=0:

            df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
        else:
            df_max_scaled[column] = df_max_scaled[column] 

    return df_max_scaled

        


train=pd.read_csv("train.csv")


result=train.label


train.drop('label', inplace=True, axis=1)

train = normalize(train)
print("input Data")
print(train)


train=train.to_numpy()

result=result.to_numpy()

train=np.transpose(train)
result=np.transpose(result)
print("shape of training data"   +      str(train.shape))
print("shape of result  data"   +      str(result.shape) )


encoded_result = to_categorical(result)
encoded_result=np.transpose(encoded_result)

print("shape of hot encoded result   data"   +      str(encoded_result.shape))               # hot encoded used for training 


import pandas as pd 

import numpy as np
from numpy import exp

np.set_printoptions(suppress=True)



class ann:



    def __init__(self,inputs , outputs ,results,hidden_layers,l):
        self.l=l 
        self.results=results 
        self.inputs=inputs
        self.outputs=outputs

        self.num_inputs =np.shape(inputs)[0]
        self.num_outputs=np.shape(outputs)[0]
        
        number_of_samples=np.shape(inputs)[1]
        self.m=number_of_samples
        
        layers=[self.num_inputs]+hidden_layers+[self.num_outputs]  
        self.layers=layers 

        weights=[]     
        bias=[]
        biases=[]  
        error_wrt_a=[]
        error_wrt_z=[]
        error_wrt_w=[]
        error_wrt_b=[]
        for i in range(len(layers)-1):            
            a=np.random.rand (layers[i+1],layers[i])   
            weights.append(a)
            
            b=np.random.rand(layers[i+1],1)                                                      # lists as well as numpy are both dynnamic we could do  appending and deleting at run times so we cannot access elements until assigned 
            bias.append(b)                                
            biases.append(np.repeat(b , repeats =self.m, axis=1)) 
            
            c=np.zeros(  ( (layers[i+1]),self.m        )     )    
            error_wrt_a.append(c)
            
            d=np.zeros(((layers[i+1]),self.m))
            error_wrt_z.append(d)
            
            e=np.zeros(((layers[i+1],layers[i])))
            error_wrt_w.append(e)
            
            f=np.zeros((layers[i+1],1))
            error_wrt_b.append(f)
            
        self.weights=weights
        self.bias=bias
        self.biases=biases
        
        self.activations=[]
        self.z=[]

        self.error_wrt_a=error_wrt_a
        self.error_wrt_z= error_wrt_z
        self.error_wrt_w= error_wrt_w
        self.error_wrt_b= error_wrt_b
        
        self.function_derivitive=[self.sigmoid_derivative,self.reluDerivative]
        self.function=[self.sigmoid,self.relu,self.softmax]
      

    def sigmoid_derivative(self, x):        
        return np.multiply(x,(1-x))

        
    def sigmoid(self , x):
        y=1/(1+np.exp(-x) )                       
        return y

    def relu(self,x):
        return np.maximum(0.0, x)
        
    
    def reluDerivative(self,x):
        y=np.copy(x)
        y[y<=0] = 0
        y[y>0] = 1
        return y                  #  important 


    def softmax (self,x):
        e=np.exp(x)
        return e/(np.sum(e,axis=0))   


    def forward_propogate(self):

        self.activations.clear()
        self.z.clear()
        self.z.append(self.inputs)
        self.activations.append(self.inputs)        ### note z and a will contain all the layers from  1st given layer at indeex 0 to the last output predicted layer 
        for i in range (len(self.layers)-1):
            self.z.append(np.add( np.dot( self.weights[i],self.activations[i] ) ,np.repeat(self.bias[i],self.m,axis=1)) )    
            self.activations.append(self.function[self.l[i]](self.z[i+1]  ) )            
    def backpropogation(self ):                                                           # for last  layer error wrt z is same in both softmax  as well as sigmoid 
       
        self.error_wrt_z[len(self.layers)-2]=np.subtract(self.activations[len(self.layers)-1] , self.outputs)
       # self.error_wrt_a[len(self.len(self.layers)-2)]=np.subtract(np.divide(1-self.outputs,self.activations[len(self.layers-1)] ) , np.divide(self.outputs,self.activations[len(self.layers-1)] )  )  # in this case we are always taking last one as sigmoid 
        for i in reversed(range(len(self.layers) - 1)):
            self.error_wrt_w[i]=1/self.m*(np.dot(self.error_wrt_z[i],np.transpose(self.activations[i])))
            self.error_wrt_b[i]=1/self.m *(np.sum(self.error_wrt_z[i] , axis=1 , keepdims=True))        # this keep dims is very important to maintain proper dimensions for b 
            if i>=1:
                self.error_wrt_z[i-1]=np.multiply (  np.dot ( np.transpose (self.weights[i]) , self.error_wrt_z[i] )  , self.function_derivitive[ self.l[i-1] ] ( self.function[ self.l[i-1] ]   (self.z[i]) ) )
            else:
                break
    def get_predictions(self):                   
        
        if self.l[-1]==2:            # its  giving single array like [1,2,0,8,9,7] etc
 
            return (np.argmax(self.activations[-1], axis=0))          # like this [1,2,3,4,4,7,0,9]  for different samples in different columns of a row

        if self.l[-1]==0:
            
            y=np.copy(self.activations[-1])
            y[y>=0.5]=1               # copying so that out finnal activation layer actually dont change 
            y[y<0.5]=0    # its giving a matrix with one row like this [  [0,0,1,0,1,1,0,0]           ]

            return (y)
            
    
    def get_accuracy(self):

        np.sum(self.get_predictions==self.results)/self.results.size






    


   



                
    
    def train(self,epoches,alpha):
        for i in range(epoches):
            self.forward_propogate()
          #  self.get_predictions()
            self.get_accuracy()
            self.backpropogation()
            for i in range(len (self.layers)-1):
                self.weights[i]=np.subtract(self.weights[i],alpha*self.error_wrt_w[i])
                self.bias[i]=np.subtract(self.bias[i],alpha*self.error_wrt_b[i])
          
            """
            for i in range (len(self.layers) - 1):
                print("---------------------------------------------ERROR WRT W ________________________________________________________________________________")
                print (self.error_wrt_w[i])
                print("------------------------------------------------ERROR WRT B ------------------------------------------------------------------------------")
                print(self.error_wrt_b[i])
            """

            
a=ann(inputs=train, outputs=encoded_result ,results=result,hidden_layers=[392,49,12],l=[1,0,1,2])


a.train(epoches=100,alpha=0.01)

    
    
  
        
    

     


 

    
    
    
            
            
            
          





                

   



