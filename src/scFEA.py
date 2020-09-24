# -*- coding: utf-8 -*-
"""

@author: wnchang@iu.edu
"""

# system lib
import argparse
import sys
import time
import os

# tools
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn


# Flux lib
from ClassFlux import FLUX # Flux class network


# hyper parameters
BATCH_SIZE = 1
LEARN_RATE = 0.008  # 0.008 for m14 ; 0.02 for m100, 0.01 for m100 162 cell.
EPOCH = 100
LAMB_BA = 1
LAMB_SCALE = 2
LAMB_SIMI = 0


def myLoss(m,  c, df, lamb1 = 0.2, lamb2= 0.2, lamb3 = 0.2, cellId = 1):    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #print(c)
    #print(m.shape)
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim = 1) #c -> 0.
#    total1 = torch.pow(total1, 0.5)
    
    df.index = [i.split('_')[0] for i in df.index]
    T = df.groupby(df.index).sum(axis=1).T
#    print(T.values[:,1])
    
       
    #module 14 constrain
#    error_a = (m[:,4] - torch.FloatTensor(T.values[:,1])).to(device)
#    error_a = torch.pow(error_a, 2)
#    error_b = (m[:,1] - torch.FloatTensor(T.values[:,1])).to(device)
#    error_b = torch.pow(error_b, 2)
#    error = error_a + error_b
    
#    #module 100 constrain
    error = (m - torch.FloatTensor(T.values)).to(device)
    error = torch.pow(error, 2)
    error = torch.sum(error, dim=1)/10
    error = torch.pow(error, 0.5)

    
    total2 = error
            
    total = lamb1 * total1 + lamb2 * total2 
    
    loss1 = torch.sum(total1)
    loss2 = torch.sum(lamb2 * total2)
    loss = torch.sum(total) 
    return loss1, loss2, loss


def main(args):
    

    
	# set arguments
    data_path = args.data_dir
    res_dir = args.res_dir
    test_file = args.test_file
    moduleGene_file = args.moduleGene_file
    cm_file = args.stoichiometry_matrix
    
    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # read data
#    os.chdir('your_current_directory')
    geneExpr = pd.read_csv(
#                "./data/Melissa_metabolic_c88_m14.csv",
				data_path + '/' + test_file,	
                index_col=0)
    geneExpr = geneExpr.T
    if geneExpr.max().max() > 30:
        geneExpr = (geneExpr + 1).apply(np.log2)  #GSE103322 no log2
    geneExprSum = geneExpr.sum(axis=1)
    stand = geneExprSum.mean()
    geneExprScale = geneExprSum / stand
    geneExprScale = torch.FloatTensor(geneExprScale.values).to(device)
    
    
    moduleGene = pd.read_csv(
#                "./data/module_gene_m14.csv",
				data_path + '/' + moduleGene_file, 
                sep=',',
                #header=None,
                index_col=0)
#    moduleGene = moduleGene.replace(np.nan, '', regex=True)
    
    cmMat = pd.read_csv(
#            "./data/cmMat_m14.csv",
			data_path + '/' + cm_file,
            sep=',',
            header=None)
    cmMat = cmMat.values
    cmMat = torch.FloatTensor(cmMat).to(device)

    
    emptyNode = []
    gene_names = geneExpr.columns
    cell_names = geneExpr.index.astype(str)
    n_modules = moduleGene.shape[0]
    n_genes = len(gene_names)
    n_cells = len(cell_names)
    n_comps = cmMat.shape[0]
    geneExprDf = pd.DataFrame(columns = ['Module_Gene'] + list(cell_names))
    for i in range(n_modules):
        genes = moduleGene.iloc[i,:].values.astype(str)
        genes = [g for g in genes if g != 'nan']
        if not genes:
            emptyNode.append(i)
            continue
        temp = geneExpr.copy()
        temp.loc[:, [g for g in gene_names if g not in genes]] = 0
        temp = temp.T
        temp['Module_Gene'] = ['%02d_%s' % (i,g) for g in gene_names]
        geneExprDf = geneExprDf.append(temp, ignore_index = True)
    geneExprDf.index = geneExprDf['Module_Gene']
    geneExprDf.drop('Module_Gene', axis = 'columns', inplace = True)
    X = geneExprDf.values.T
    X = torch.FloatTensor(X).to(device)

    
    
# =============================================================================
    #NN
    torch.manual_seed(16)
    net = FLUX(X, n_modules, f_in = n_genes, f_out = 1).to(device)
#    print(net.m_encoder[0].parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr = LEARN_RATE)
#    optimizer = torch.optim.Adam(net.m_encoder[0].parameters(), lr = LEARN_RATE)
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    #Dataloader
    dataloader_params = {'batch_size': BATCH_SIZE,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(
                                                torch.utils.data.TensorDataset(X),
                                                **dataloader_params)
    
# =============================================================================

  
    
    # initialize
#    def init_weights(w):
#        if type(w) == torch.nn.Linear:
#            torch.nn.init.xavier_uniform(w.weight, gain = 10)
#            w.bias.data.fill_(0.01)
#    net.apply(init_weights)
   
    
# =============================================================================
    start = time.time()  
#     training
    lamb_ba = LAMB_BA
    lamb_scale = LAMB_SCALE
    lamb_simi = LAMB_SIMI
    loss_v = []
    loss_v1 = []
    loss_v2 = []
    net.train()
    for epoch in range(EPOCH):
        loss, loss1, loss2 = 0,0,0
        
        for i, item in enumerate(train_loader): # 162
            
            X_batch = Variable(item[0])
            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat, geneExprScale, i)

#            if i == 0:
#                out_m = out_m_batch
#                out_c = out_c_batch
#            else:
#                out_m = torch.cat((out_m, out_m_batch), dim = 0)
#                out_c = torch.cat((out_c, out_c_batch), dim = 0)
               
            X_batch_df = pd.DataFrame(X_batch.cpu().data.numpy().T, index = geneExprDf.index)
            loss1_batch, loss2_batch, loss_batch = myLoss(out_m_batch, out_c_batch, X_batch_df, lamb1 = lamb_ba, lamb2 = lamb_scale, lamb3 = lamb_simi, cellId = i)
            # back propagation
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
            loss += loss_batch.cpu().data.numpy()
            loss1 += loss1_batch.cpu().data.numpy()
            loss2 += loss2_batch.cpu().data.numpy()
            
#        scheduler.step(loss)
#        print("m...", out_m_batch[0,:])
#        print("c*********", out_c_batch[0,:])
#        print('epoch: %02d, loss1: %.8f, loss2_lambda_free: %.8f, loss: %.8f' % (epoch+1, loss1, loss2/lamb_scale, loss))
        if(epoch % 5 == 0):
            print('epoch:', epoch)
        loss_v.append(loss)
        loss_v1.append(loss1)
        loss_v2.append(loss2/lamb_scale)
        
# =============================================================================
    end = time.time()
    print("Training time: ", end - start) 
    
    plt.plot(loss_v)
    plt.plot(loss_v1)
    plt.plot(loss_v2)
    plt.legend(['total', 'balance', 'scale']);
    timestr = time.strftime("%Y%m%d-%H%M%S")
#    imgName = "./output/loss_" + timestr + ".png"
    imgName = './' + res_dir + '/loss_' + timestr + ".png"
    plt.savefig(imgName)
    
    #save model
#    torch.save(net, './output/model_Melissa_m14_c88' + timestr + '.pkl')
    
    
#    Dataloader
    dataloader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}
    test_loader = torch.utils.data.DataLoader(
                                                torch.utils.data.TensorDataset(X),
                                                **dataloader_params)
   
    #testing
    fluxStatuTest = np.zeros((n_cells, n_modules), dtype='f') #float32
    net.eval()
    for epoch in range(1):
        loss, loss1, loss2 = 0,0,0
        for i, item in enumerate(test_loader): # 162
            #print(i)
            #optimizer.zero_grad()
            X_batch = Variable(item[0])
            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat, geneExprScale, i)
            
            # save data
            fluxStatuTest[i, :] = out_m_batch.detach().numpy()
            
            
    
    # save to file
    fileName = "./" + res_dir + "/module" + str(n_modules) + "_cell" + str(n_cells) + "_batch" + str(BATCH_SIZE) + "_LR" + str(LEARN_RATE) + "_lambSca" + str(LAMB_SCALE) + "_" + timestr + ".csv"
    setF = pd.DataFrame(fluxStatuTest)
    setF.columns = moduleGene.index
    setF.index = geneExpr.index.tolist()
    setF.to_csv(fileName)
    
    print("Done. Check result in the desired output folder.")
    
    return
	
def parse_arguments(parser):


    parser.add_argument('--data_dir', type=str, default='data', metavar='<data_directory>',
                        help='The data directory for input data')
    parser.add_argument('--res_dir', type=str, default='output', metavar='<data_directory>',
                        help='The data directory for result [output]')
    parser.add_argument('--test_file', type=str, default='Melissa_metabolic_c88_m14.csv', help='The test SC file [input]')
    parser.add_argument('--moduleGene_file', type=str, default='module_gene_m14.csv', 
						help='The table contains genes for each module.')
    parser.add_argument('--stoichiometry_matrix', type=str, default='cmMat_m14.csv', 
						help='The table descript relationship between compounds and modules.')


    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='scFEA: A graph neural network model to estimate cell-wise metabolic using single cell RNA-seq data')
    args = parse_arguments(parser)
    main(args)
