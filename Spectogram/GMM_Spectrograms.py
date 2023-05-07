import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import silence_tensorflow.auto
from tqdm import tqdm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

os.chdir('C:/Users/Timothy/Documents/GitHub/COMPSYS-ELECTENG-700/Spectogram')

def em(data, components, iterations, RNGseed):
 Number_of_samples = data.shape[0]
 np.random.seed(RNGseed)
 
 mean = np.random.rand(components)
 variance = np.random.rand(components)
 components_probs = np.random.dirichlet(np.ones(components))
 
 for em in range(iterations):
  #responsibilities = np.zeros((Number_of_samples,components))
  #for i in range(Number_of_samples):
   #for component in range (components):
    #responsibilities[i,component] = components_probs[c] *\
    #tfp.distributions.distributions.Normal(loc= mean[component],scale = variance[component].prob(data[i]))
  responsibilities=tfp.distributions.Normal(loc=mean, scale = variance).prob(data.reshape(-1,1)).numpy()
  responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)
  components_responsibilities = np.sum(responsibilities, axis=0)
 
  for component in range(components):
   components_probs[component] = components_responsibilities[component]/Number_of_samples
   mean[component] = np.sum(responsibilities[:,component]* data, )/components_responsibilities[component]
   variance[component] = np.sqrt(np.sum(responsibilities[:,component]*(data-mean[component])**2))/components_responsibilities[component]
 return mean, variance, components_probs

img_files = os.listdir('Spectrogram_Plots/Mag')
n=len(img_files)
#True clusters
N_clusters = 18
iterations = 100
#RNG seed
seed = 21
#Reds = np.zeros()
#Blues = np.zeros()
#Greens = np.zeros()
for i in range (0,len(img_files)):
 #To be fixed, FUcking tuples for images.
 #RGB values (y, x, color)
 img=mpimg.imread('Spectrogram_Plots/Mag/'+ img_files[i])
 Reds = img[:,:,1]
 Blues= img[:,:,2]
 Greens= img[:,:,3] 
#Red_mean, Red_variance, Rcomponent_probs = em(Reds, N_clusters, iterations,seed) 
#Blue_mean, Blue_variance, Bcomponent_probs  = em(Blues,N_clusters, iterations,seed)
#Green_mean, Green_variance, Gcomponent_probs = em(Greens, N_clusters, iterations,seed)
print(np.asarray(Reds))
#print(Red_mean)
#print(Red_variance)
#print(Rcomponent_probs)


#print(img)
    #plt.imshow(img)
    #plt.show()
    #plt.close()
#if __name__ == "__main__":
  #  main()