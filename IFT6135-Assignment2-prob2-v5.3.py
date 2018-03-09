# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:54:54 2018

@author: Moneim
"""

import cv2
import time
import pickle
import urllib
from urllib.request import urlretrieve
urlretrieve
#import cPickle as pickle
from random import shuffle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.utils.data
import torch.utils.data.sampler  as sampler
import torch.optim as optim
import gzip
import os
import numpy as np
import zipfile
import scipy.ndimage
import glob
from glob import glob
from scipy.ndimage import imread
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns
import shutil
import warnings
warnings.filterwarnings("ignore")



model_path='output'
model_interv = 5


#%%
# Load the dog_vs_cat data
CUDA = torch.cuda.is_available()
os.chdir("/Users/Moneim/Desktop/Umontreal/ML Master/IFT6135/Assignment2")
print(os.getcwd())

DC_TRAIN = '/Users/Moneim/Desktop/Umontreal/ML Master/IFT6135/Assignment2/dataset/train_64x64'
DC_VALID = '/Users/Moneim/Desktop/Umontreal/ML Master/IFT6135/Assignment2/dataset/valid_64x64'


def parse_jpegs(files):
    mats = []
    labels = []

    for d in files:
        a = imread(d)

        if 'Cat' in d:
            labels.append(0)
        else:
            labels.append(1)

        # we have some greyscale images with no color channel, so we expand
        # it (duplicate) so we do...
        if len(a.shape) != 3:
            a = np.repeat(np.expand_dims(a, axis=-1), 3, axis=-1)

        mats.append(a.T)

    mats = np.stack(mats, axis=0)
    labels = np.array(labels)

    return(mats.astype(np.float), labels)


def load_dc(batch_size=64):
    """Loads cat and dog data into train / valid / test dataloders"""
    train_files = glob(os.path.join(DC_TRAIN, '*.jpg'))
    valid_files = glob(os.path.join(DC_VALID, '*.jpg'))

    train_mats, train_labels = parse_jpegs(train_files)
    valid_mats, valid_labels = parse_jpegs(valid_files)

    means = [np.mean(train_mats[:,0,:,:]),
             np.mean(train_mats[:,1,:,:]),
             np.mean(train_mats[:,2,:,:])]

    train_mats[:,0,:,:] -= means[0]
    train_mats[:,1,:,:] -= means[1]
    train_mats[:,2,:,:] -= means[2]
    valid_mats[:,0,:,:] -= means[0]
    valid_mats[:,1,:,:] -= means[1]
    valid_mats[:,2,:,:] -= means[2]

    valid_cutoff = valid_mats.shape[0] // 2

    test_mats = valid_mats[valid_cutoff:, :, :, :]
    valid_mats = valid_mats[:valid_cutoff, :, :, :]
    test_labels = valid_labels[valid_cutoff:]
    valid_labels = valid_labels[:valid_cutoff]

    # convert
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_mats), torch.ByteTensor(train_labels))
    valid_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(valid_mats), torch.ByteTensor(valid_labels))
    test_dataset  = torch.utils.data.TensorDataset(torch.FloatTensor(test_mats),  torch.ByteTensor(test_labels))

    # loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True, num_workers=2)

    loaders = [train_loader, valid_loader, test_loader]

    return(loaders)

 


#%%
class  ConvBatchRelu(nn.Module):
    def __init__(self,dim_in,dim_out,kernel=3,stride=1):
        super(ConvBatchRelu,self).__init__()
        
        padding=int((kernel-1)/2)
        #define conv layer with Batchnorm and Relu as activation
        self.conv=nn.Conv2d(in_channels=dim_in,out_channels=dim_out,kernel_size=kernel,
                       padding=padding,stride=stride,bias=False)
        self.batch_norm=nn.BatchNorm2d(dim_out)
        self.relu_fc=nn.ReLU(inplace=True)
        
    def forward(self,x):
        out=self.relu_fc(self.batch_norm(self.conv(x)))
        
        return out

#%%
class  ConvBatch(nn.Module):
    def __init__(self,dim_in,dim_out,kernel=3,stride=1):
        super(ConvBatch,self).__init__()
        
        padding=int((kernel-1)/2)
        #define conv layer with Batchnorm and Relu as activation
        self.conv=nn.Conv2d(in_channels=dim_in,out_channels=dim_out,kernel_size=kernel,
                       padding=padding,stride=stride,bias=False)
        self.batch_norm=nn.BatchNorm2d(dim_out)
                
    def forward(self,x):
        out=self.batch_norm(self.conv(x))
        
        return out

#%%
class residualBlock(nn.Module):
    def __init__(self,dim_out):
        super(residualBlock,self).__init__()
        
        self.layer1=ConvBatchRelu(dim_out,dim_out)
        self.layer2=ConvBatch(dim_out,dim_out)
        
        self.relu=nn.ReLU(inplace=True)
                
        
    def forward(self,x):
        residual=x
        
        out=self.layer1(x)
        out=self.layer2(out)
        out+=residual
        out=self.relu(out)
        
        return out
    
#%%
class residualNet(nn.Module):
    def __init__(self,dim_in,dim_out,kernel=3,stride=1,padding=1,blocks=4):
        super(residualNet,self).__init__()
        
        
        #build conv layers with Resnet method
        convRes=[ConvBatchRelu(dim_in,64,kernel=1)]
        convRes.append(residualBlock(64))                         ##64x64 
        
        convRes.append(ConvBatchRelu(64,128,kernel=5,stride=2))
        convRes.append(residualBlock(128))                        ##32x32
        
        convRes.append(ConvBatchRelu(128,256,kernel=7,stride=2))
        convRes.append(residualBlock(256))                        ##16x16
        
        convRes.append(ConvBatchRelu(256,512,stride=1))           ##16x16
        
        if int(blocks)>2:
            for i in range(int(blocks)-2):
                convRes.append(residualBlock(512))                ##16x16
        
        convRes.append(nn.MaxPool2d(2,stride=2)) 
                
        
        #build fully connected layers
        fc_lin=[nn.Linear(32768,4096),nn.BatchNorm2d(4096),nn.ReLU()]
        fc_lin.append(nn.Linear(4096,4096))
        fc_lin.append(nn.BatchNorm2d(4096))
        fc_lin.append(nn.ReLU())
        
        fc_lin.append(nn.Linear(4096,1000))
        fc_lin.append(nn.BatchNorm2d(1000))
        fc_lin.append(nn.ReLU())
        
        fc_lin.append(nn.Linear(1000,2))                              #16x16x512 
        
        #finalize the architectures
        self.cnn=torch.nn.Sequential(*convRes)
        self.fc=torch.nn.Sequential(*fc_lin)
        self.clf=torch.nn.LogSoftmax(dim=0)
        
        
        
    def forward(self,x):
        
        x = self.cnn(x)
        dims = x.shape
        x = x.view(dims[1]*dims[2]*dims[3], -1).transpose(0, 1)
        
        return(self.fc(x))
        
    
        
#%%                
    def initalizer(self, init_type='glorot'):
        """
        Takes in a model, initializes it to all-zero, normal distribution
        sampled, or glorot initialization. Golorot == xavier.
        """
        if init_type not in ['zero', 'normal', 'glorot']:
            raise Exception('init_type invalid]')

        for k, v in self.cnn_A.named_parameters():
            if k.endswith('weight'):
                if init_type == 'zero':
                    torch.nn.init.constant(v, 0)
                elif init_type == 'normal':
                    torch.nn.init.normal(v)
                elif init_type == 'glorot':
                    torch.nn.init.xavier_uniform(v, gain=calculate_gain('relu'))
                else:
                    raise Exception('invalid init_type')
#%%
    def count_params(self):
        """
        Returns a count of all parameters
        """
        param_count = 0
        for k, v in self.cnn.named_parameters():
            param_count += np.prod(np.array(v.size()))

        return(param_count)
    
    
#%%
    def adjust_lr(self, optimizer, epoch, total_epochs):
        # source : https://github.com/sukilau/Ziff-deep-learning/blob/master/3-CIFAR10-lrate/CIFAR10-lrate.ipynb
      
        lr = lr_0* (0.1 ** (epoch / float(total_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
 
    
    


#%%
    def predict(self, x):
        return (self.clf(x))


#%%
    def save_model(self, ep):
        torch.save(self, '%s/modela_save_%d' % (model_path, ep))
        
        
   
#%%
        
        
#%%
        
def evaluate(model, dset,dropout=False,convmode=True):
    # Evaluate,This has any effect only on modules such as Dropout or BatchNorm.
    total = 0
    correct = 0
    for i, (imgs, label) in enumerate(dset):
        if cuda_available:
            imgs, label = imgs.cuda(), label.cuda()

        imgs, label = Variable(imgs, volatile=True), Variable(label, volatile=True)
        
        if convmode:
            if len(imgs.shape) == 3:
                outputs = model.forward(imgs.unsqueeze(1)) # because no channel dimention for mnist
            else:
                outputs = model.forward(imgs)
        else:
            outputs = model.forward(imgs.view(imgs.shape[0], -1))
        
        if dropout:
            outputs=outputs*0.5
        
        output_pred=model.predict(outputs)
        
        _, predicted = torch.max(output_pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()
        #print(total,correct)
    return 100.0*correct/total



#%% 
#evealuate the accuracy of running the model over train,valid,test data sets

def run_experiment(model,loaders,weight_decay,lr_0,num_epochs,dset='train',convmode=True,dropout=False,momentum=False, display=3):
    
    if len(loaders) == 3:
        train, valid, test = loaders
    elif len(loaders) == 2:
        train, valid = loaders
        test = None
    else:
        raise Exception('loaders malformed')
    
    if dset=='train':
        dset=train
    else:
        dset=test
        
    model.initalizer(init_type='glorot')
    
    

    if cuda_available:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_0, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=2)
    
    
    epoch_losses=[]  
    epoch_losses,train_acc_l , valid_acc_l = [],[],[]
    best_valid_acc,gen_gap=0,0
    
    for ep in range(num_epochs):
        # This has any effect only on modules such as Dropout or BatchNorm.
        #self.train()
        
        losses = []
        output_pred_l=[]
        # Train
        start = time.time()
        # for batch_idx, (inputs, targets) in enumerate(tqdm(train_data_loader)):
        for i, (imgs, label) in enumerate(dset):
            
            if cuda_available:
                imgs, label = imgs.cuda(), label.cuda()
                
            # img = transforms.ToPILImage(inputs)
            imgs, label = Variable(imgs, volatile=True), Variable(label, volatile=True)
                
            optimizer.zero_grad()
                
            if convmode:
                if len(imgs.shape) == 3:
                    outputs = model.forward(imgs.unsqueeze(1)) # because no channel dimention for mnist
                else:
                    outputs = model.forward(imgs)
            else:
               outputs = model.forward(imgs.view(imgs.shape[0], -1))
            
            if dropout:
                outputs=outputs*0.5
             
                
            output_pred=model.predict(outputs)
            output_pred_l.append(output_pred)
            
            loss = criterion(output_pred, label)
            loss.backward()
            scheduler.step()
            optimizer.step()
            losses.append(loss.data[0])
        end = time.time()
            
        # print the process of the training.  
        # if display!=0 and ep % display==0:
        #     print('Epoch : %d Loss : %.3f Time : %.3f seconds ' % (ep, np.mean(losses), end - start))
            
        # Evaluate,This has any effect only on modules such as Dropout or BatchNorm.
        #self.eval()
            
        epoch_losses.append(np.mean(losses))
            
        train_acc=evaluate(model,train,dropout,convmode)
        train_acc_l.append(train_acc)
        valid_acc=evaluate(model,valid,dropout,convmode)
        valid_acc_l.append(valid_acc)
            
        if valid_acc>best_valid_acc:
            best_valid_acc=valid_acc
            if test:
                test_acc=evaluate(model,test,dropout,convmode)
                gen_gap=train_acc-test_acc
            
        # print the process of the training.  
        if display!=0 and (ep+1) % display==0:
            cur_lr=optimizer.state_dict()['param_groups'][0]['lr']
            print('+ [{:03d}] Epoch_losses={:0.6f} Train_Acc={:0.2f} Valid_Acc={:0.2f} cur_lr={} Spend={:0.2f} minutes'.format(
                    (ep+1),epoch_losses[-1], train_acc, valid_acc,cur_lr,(end-start)/60.0))
            print('--------------------------------------------------------------')

        #if ep % model_interv == 0:
        #   model.save_model(ep)
        
        
    results={'Epoch_losses':epoch_losses,
             'train_acc':train_acc_l,
             'valid_acc':valid_acc_l,
             'best_valid_acc':best_valid_acc,
             'gen_gap':gen_gap,
             'predict':output_pred_l}
        
    return results




#%% plot count of cats and dogs
train_files = glob(os.path.join(DC_TRAIN, '*.jpg'))
valid_files = glob(os.path.join(DC_VALID, '*.jpg'))

    
labels = []
for i in train_files:
    if 'Dog' in i:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels)
plt.grid()
plt.title('Cats and Dogs')
plt.show()

#%%
#Define function to plot model accuracy
def plot_acc(result):
    
    fig = plt.figure()
    plt.plot(range(1,num_epochs-1),result['valid_acc'],label='validation')
    plt.plot(range(1,num_epochs-1),result['train_acc'],label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1,num_epochs])
#     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    #fig.savefig('img/'+str(i)+'-accuracy.jpg')
    #plt.close(fig)     
    
#%%
#%%
#Define function to plot model loss
def plot_loss(result_train,result_val):
    
    fig = plt.figure()
    plt.plot(range(1,num_epochs-1),result_train['Epoch_losses'],label='val_loss')
    plt.plot(range(1,num_epochs-1),result_val['Epoch_losses'],label='train_loss')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim([1,num_epochs])
#     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Error")
    plt.show()
    #fig.savefig('img/'+str(i)+'-accuracy.jpg')
    #plt.close(fig)     
    

#%% define function to visualize misclassifications
def miss_calss(results):
    
    for i in range(0,10):
        if result['predict'][i] >= 0.5:
            print('I am {:.2%} sure this is a Dog'.format(result['predict'][i]))
        else:
            print('I am {:.2%} sure this is a Cat'.format(1-result['predict'][i]))
        
    plt.imshow(train[10].T)
    plt.show()
    
    

  
#%%
'''
Train Model 1
'''

# hyper parameters of Deep learning
num_epochs = 5
lr_0=1e-2
momentum=0.9
weight_decay=5e-4

cuda_available=False

model=residualNet(3,64)

#load Data
loaders = load_dc(batch_size=8)

#show no# of parameters
residualNet.count_params(model)

#run experiment
results_train=run_experiment(model,loaders,weight_decay,lr_0,num_epochs,'train',momentum=0.9, display=3)
results_test=run_experiment(model,loaders,weight_decay,lr_0,num_epochs,'test',momentum=0.9, display=3)

#plot acc
plot_acc(results_train)

#plot loss
plot_loss(3,results_train,results_test)


    
    
#%%        
    

'''
model=residualNet(3,64)
model=DeepBoi()
loaders=load_img()
model.predict(imgs)
dropout=False
convmode=True
total = 0
correct = 0
for i, (imgs, label) in enumerate(train):
    if cuda_available:
       imgs, label = imgs.cuda(), label.cuda()

    imgs, label = Variable(imgs, volatile=True), Variable(label, volatile=True)
    #print(imgs.shape)
    if convmode:
        if len(imgs.shape) == 3:
            outputs = model.forward(imgs.unsqueeze(1)) # because no channel dimention for mnist
        else:
            outputs = model.forward(imgs)
    else:
        outputs = model.forward(imgs.view(imgs.shape[0], -1))
        
    if dropout:
        outputs=outputs*0.5
        
    output_pred=model.predict(outputs)
        
    _, predicted = torch.max(output_pred.data, 1)
    total += label.size(0)
    correct += predicted.eq(label.data).cpu().sum()
    print(correct/total)
    
    


total = 0
correct = 0
for i, (imgs, label) in enumerate(train):
    if cuda_available:
       imgs, label = imgs.cuda(), label.cuda()

    imgs, label = Variable(imgs, volatile=True), Variable(label, volatile=True)
        
    outputs = model.forward(imgs)
    _, predicted = torch.max(outputs.data, 1)
    total += label.size(0)
    correct += predicted.eq(label.data).cpu().sum()
    print(correct/total)
'''