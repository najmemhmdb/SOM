##########################################################################################
#najmeh mohammadbagheri
#99131009
#najmeh.mohammadbagheri77@gmail.com
##########################################################################################



import os
import numpy as np
import matplotlib.pyplot as plt
import math
##########################################################################################
def load_dataset():
    print(os.getcwd())
    os.chdir('/content/drive/My Drive/ANN/hw3/yalefaces')
    print(os.getcwd())
    listOfFiles = os.listdir('.')
    X = []
    y = []

    for filename in listOfFiles:
        X.append(np.ndarray.flatten(plt.imread(filename)))
        y.append(filename.split('.')[1])
    X = np.array(X) / 255
    os.chdir('/content/drive/My Drive/ANN/hw3')
    print(X.shape)
    return X,y
##########################################################################################
# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
# print(tf.__version__)

try:
#   %tensorflow_version 2.x
  
#   %load_ext tensorboard
except:
  pass

from tensorflow import keras
##########################################################################################
class EarlyStoppingCallback(keras.callbacks.Callback):
  def __init__(self,patience=0):
    super(EarlyStoppingCallback,self).__init__()
    self.patience = patience
    # self.val_loss = []
    # self.val_acc = []
    self.train_loss = []
    self.train_acc = []
  def on_train_begin(self,logs=None):
    self.best = -np.Inf
    self.wait  = 0
    self.stopped_epoch = 0

  def on_epoch_end(self,epoch,logs =None):
    self.train_acc.append(logs.get("accuracy"))
    self.train_loss.append(logs.get("loss"))
    # print(logs)
    self.stopped_epoch = epoch
    current_acc = logs.get("accuracy")
    print(current_acc)
    if np.greater(current_acc,self.best):
      self.best = current_acc
      print(self.best)
      print("************")
      self.best_weights = self.model.get_weights()
  

  def on_train_end(self,logs = None):
    x = list(range(self.stopped_epoch))
    plt.plot(x,self.train_loss[:len(x)],color='orange',label='train')
    # plt.plot(x,self.val_loss[:len(x)],color='b',label='val')
    plt.title("epoch_loss")
    plt.legend()
    plt.show()
    plt.plot(x,self.train_acc[:len(x)],color='orange',label="train")
    # plt.plot(x,self.val_acc[:len(x)],color='b',label="val")
    plt.title('epoch_accuracy')
    plt.legend()
    plt.show()
    self.model.set_weights(self.best_weights)
    if self.stopped_epoch > 0 :
      print("epoch: %d: early stopping" %(self.stopped_epoch))
      print(self.best)
##########################################################################################
model = keras.models.Sequential([
                                 keras.layers.Dense(1000,input_shape=(25,),name='input_layer'),
                                 keras.layers.Dense(units=500,activation='relu',name='hidden_layer_1'),
                                #  keras.layers.Dense(units=500,activation='relu',name='hidden_layer_2'),
                                 keras.layers.Dense(units=100,activation='relu',name='hidden_layer_3'),
                                 keras.layers.Dense(11, name= 'output_layer',activation='softmax')
])
# keras.utils.plot_model(model,show_shapes= True, show_layer_names= True, expand_nested=True)
####################################################################################################################################################################################
class SOM:
    def __init__(self,map_size,R0 = 3,lr=0.1):
        """

        :param map_size: [map_w, map_h, f= 77760]
        :param lr: learning rate
        """
        # self.colors_dict = {'centerlight':[255,255,0] , 'glasses':[0,255,255] ,'happy':[255,0,0],
        # 'leftlight':[0,0,255], 'noglasses':[0,255,0] ,'normal':[128,0,128],
        # 'rightlight':[150,50,100],'sad':[0,128,128],'sleepy':[192,192,192] ,'surprised':[0,0,139],'wink':[219,112,147]}
        self.colors_dict = {0:[255,255,0] ,1:[0,255,255] ,2:[255,0,0],3:[250,100,0], 4:[0,255,0] ,5:[150,10,100],6:[105,50,10],
                            7:[0,128,128],8:[192,192,192] ,9:[0,0,139],10:[219,112,147]}
        self.color_legend = {'centerlight': 'yellow' , 'glasses':'sky blue' ,'happy':'red',
        'leftlight':'blue', 'noglasses':'green','normal':'purple',
        'rightlight':'white??','sad':'teal','sleepy':'silver' ,'surprised':'dark blue','wink':'pale violet red'}
        self.label_index = {'centerlight': 0 , 'glasses':1 ,'happy':2,
        'leftlight':3, 'noglasses':4,'normal':5,
        'rightlight':6,'sad':7,'sleepy':8 ,'surprised':9,'wink':10}
        self.map = np.random.random(size=(map_size[0],map_size[1],map_size[2]))

        self.extracted_features = []

        self.scores = np.zeros([self.map.shape[0],self.map.shape[1],3])
        self.counter = np.zeros([self.map.shape[0],self.map.shape[1],11])

        self.lr0 = lr
        self.lr = self.lr0
        self.R0 = R0
        self.R = self.R0


    def train(self, X, T = 1000,error_threshold= 10**-5):
        Js = []
        for t in range(T):
            prev_map = self.map
            shuffle_ind = np.random.randint(low=0,high=len(X),size=len(X))
            for i in range(len(X)):
                x = X[shuffle_ind[i],:]
                winner = self.find_winner(x)
                n_mask = self.get_neigh_mask(winner)
                self.update_weights(x,n_mask,len(X))
            self.lr = self.lr0 * (1-t /T)
            self.R = self.R0 * (1-t /T)
            # self.lr = self.lr0 * math.exp(-t /T)
            # self.R = self.R0 * math.exp(-t/T)
            Js.append(np.linalg.norm(prev_map - self.map))
            if t % 10 == 0:
                print("Iteration: %d, LR: %f, R: %f, Js: %f " %(t,self.lr,self.R,Js[-1]))
            if Js[-1] < error_threshold:
                print("Min Change")
                print(Js[-1])
                break
            if t > 400:
                self.lr0 = 0.01
                # self.R0 = 0.01
        self.save_weights()
        return Js

    def visualize(self,X,y): 
        T = 0   
        self.counter = np.zeros([self.map.shape[0],self.map.shape[1],11])
        for i in range(len(X)):
            x = X[i,:]
            winner = self.find_winner(x)
            iw,jw = winner[0],winner[1]
            label = self.label_index.get(y[i])
            self.counter[iw,jw,label] += 1
        for r in range(self.map.shape[0]):
            for c in range(self.map.shape[1]):
                ind = np.argmax(self.counter[r,c,:])
                T += max(self.counter[r,c,:])
                if self.counter[r,c,ind] == 0:
                    self.scores[r,c] = np.asarray([0, 0, 0])
                else: 
                    self.scores[r,c] = self.colors_dict.get(ind)
        plt.imshow(self.scores.astype(np.int64))
        plt.show()
        print("purity is : %f" %(T/len(X)))


    def visualization2(self):
        fig, ax = plt.subplots(1,11)
        fig.set_size_inches(22, 2)
        for i in range(11):
            ax[i].imshow(self.counter[:,:,i],aspect='auto',cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(list(self.label_index)[i])
        plt.show()



    def find_winner(self, x):
        x_rep = np.tile(x,[self.map.shape[0],self.map.shape[1],1])
        dists = np.sum((self.map - x_rep)**2,axis=2)
        winner = np.unravel_index(np.argmin(dists,axis=None),dists.shape)

        return winner

    def get_neigh_mask(self, winner):
        net_mask = np.zeros([self.map.shape[0],self.map.shape[1]])
        iw,jw = winner[0],winner[1]
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                    net_mask[i,j] = math.exp((-1 * (np.linalg.norm(np.array([i,j]) - winner)**2)) /(2*self.R**2))
        return net_mask
    
    def update_weights(self, x, n_mask,X_len):
        NS = np.tile(n_mask,[self.map.shape[2],1,1]).transpose()
        x_rep = np.tile(x, [self.map.shape[0], self.map.shape[1], 1])
        delta = x_rep - self.map
        self.map = self.map + self.lr * np.multiply(NS, delta)

    def save_weights(self):
        print(self.map[0,2,10])
        arr_reshaped = self.map.reshape(self.map.shape[0], -1)
        os.chdir('/content/drive/My Drive/ANN/hw3')
        print(os.getcwd())
        np.savetxt("weights_66.txt",arr_reshaped)


    def set_map_value(self,map_value):
        self.map = map_value

    def extract_features_all(self,X):
        new_X = []
        for x in X:
            new_X.append(self.extract_features(x))
        return np.array(new_X)

    def visualize_features(self, features, y):
        imgs = np.zeros(shape=(11,len(features[0]),len(features[0,0]))).astype('float64')
        fig, ax = plt.subplots(1, 11)
        fig.set_size_inches(20, 2)
        for i in range(len(y)):
            imgs[self.label_index.get(y[i])] += np.array(features[i])
        for j in range(11):
            ax[j].imshow(imgs[j,:,:])
            ax[j].axis('off')
            ax[j].set_title(list(self.label_index)[j])
        plt.show()

    def extract_features(self,x):
        x_rep = np.tile(x, [self.map.shape[0], self.map.shape[1], 1])
        dists = np.sum((self.map - x_rep) ** 2, axis=2)
        features = 1/(1+dists)
        self.extracted_features.append(features.flatten())
        return features

    def save_extracted_features(self):
        np.savetxt("extracted_features_t17.txt",np.array(self.extracted_features))
        return
        

label_index = {'centerlight': 0 , 'glasses':1 ,'happy':2,
        'leftlight':3, 'noglasses':4,'normal':5,
        'rightlight':6,'sad':7,'sleepy':8 ,'surprised':9,'wink':10}

if __name__ == '__main__':
    # load data and split to test and train 
    # 
    # 
    X, y = load_dataset()
    X_train = X[:154,:]
    X_test = X[154:,:]
    # y_train = y[:154]
    # y_test = y[154:]

    # part c
    # 
    # 
    # train SOM for clustering 
    # som_net = SOM(map_size=[6,6,X.shape[1]])
    # Js = som_net.train(X_train,T=600)
    # plt.plot(Js)
    # plt.show()
    # som_net.visualize(X_train,y_train)
    # som_net.visualization2()

    # test data
    # 
    #  
    # print("test")
    # som_net.visualize(X_test,y_test)
    # som_net.visualization2()


    # load saved_weights for part feature extraction
    # 
    # os.chdir('/content/drive/My Drive/ANN/hw3')
    # loaded_arr = np.loadtxt("weights_66_r2.txt")
    # load_original_arr = loaded_arr.reshape(
    #     loaded_arr.shape[0], loaded_arr.shape[1] // 77760, 77760)

    # part d   
    # 
    # feature extracetion and saveing features
    # som_net = SOM(map_size=[6,6,77760])
    # som_net.set_map_value(load_original_arr)
    # extracted_features = som_net.extract_features_all(X)
    # som_net.save_extracted_features()
    # som_net.visualize_features(extracted_features,y)

    # part e 
    # 
    # 
    # train classifire on row features 
    # 
    # 

    numerical_y = []
    for label in y:
        numerical_y.append(label_index.get(label))
    numerical_y_train = numerical_y[:154]
    numerical_y_test = numerical_y[154:]
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # er_cal = EarlyStoppingCallback(patience=10)
    # model.fit(np.array(X_train),np.array(numerical_y_train),epochs=1000,callbacks = [er_cal])
    # test_loss,test_acc = model.evaluate(X_test,np.array(numerical_y_test),verbose=2)

    # load extracted_features
    os.chdir('/content/drive/My Drive/ANN/hw3')
    new_X = np.loadtxt("extracted_features_t17.txt")
    # 
    # 
    # # train model for classification new_X
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    er_cal = EarlyStoppingCallback(patience=0)
    model.fit(new_X[:154,:],np.array(numerical_y_train),epochs=1000,callbacks = [er_cal])
    test_loss,test_acc = model.evaluate(new_X[154:,:],np.array(numerical_y_test),verbose=2)


####################################################################################################################################################################################

# plot table of colors 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# helper function to plot a color table
def colortable(colors, title, colors_sort = True, emptycols = 0):
 # cell dimensions
 width = 212
 height = 22
 swatch_width = 48
 margin = 12
 topmargin = 40
 # Sorting colors bbased on hue, saturation,
 # value and name.
 if colors_sort is True:
  to_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
      name)
      for name, color in colors.items())
  names = [name for hsv, name in to_hsv]
 else:
  names = list(colors)
 length_of_names = len(names)
 length_cols = 4 - emptycols
 length_rows = length_of_names // length_cols + int(length_of_names % length_cols > 0)
 width2 = width * 4 + 2 * margin
 height2 = height * length_rows + margin + topmargin
 dpi = 72
 figure, axes = plt.subplots(figsize =(width2 / dpi, height2 / dpi), dpi = dpi)
 figure.subplots_adjust(margin / width2, margin / height2,
      (width2-margin)/width2, (height2-topmargin)/height2)
 axes.set_xlim(0, width * 4)
 axes.set_ylim(height * (length_rows-0.5), -height / 2.)
 axes.yaxis.set_visible(False)
 axes.xaxis.set_visible(False)
 axes.set_axis_off()
 axes.set_title(title, fontsize = 24, loc ="left", pad = 10)
 for i, name in enumerate(names):
  rows = i % length_rows
  cols = i // length_rows
  y = rows * height
  swatch_start_x = width * cols
  swatch_end_x = width * cols + swatch_width
  text_pos_x = width * cols + swatch_width + 7
  axes.text(text_pos_x, y, name, fontsize = 14,
    horizontalalignment ='left',
    verticalalignment ='center')
  axes.hlines(y, swatch_start_x, swatch_end_x,
    color = colors[name], linewidth = 18)
 return figure
colortable(colors = {
           "sad": np.array([0,128,128])/255,
          "sleepy":  np.array([192,192,192])/255,
           "wink":  np.array([219,112,147])/255,
           "glasses":  np.array([0,255,255])/255,
            "noglasses":  np.array([0,255,0])/255,
            "centerlight":  np.array([255,255,0])/255,
            "rightlight":  np.array([105,50,10])/255,
            "leftlight":  np.array([250,100,0])/255,
           "happy":  np.array([255,0,0])/255,
           "normal":  np.array([150,10,100])/255,
            "surprised":  np.array([0,0,139])/255
        }, title="labels",
    colors_sort = False, emptycols = 1)
plt.show()
##########################################################################################
