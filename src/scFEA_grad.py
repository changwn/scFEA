# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:13:30 2021

@author: wnchang
"""

# system lib
#import sys
import argparse
import time
import warnings
import pickle


# tools
import torch
from torch.autograd import Variable
#import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#from torch.utils.data import DataLoader
#import torch.nn as nn
import magic
from tqdm import tqdm

# my lib
from ClassFlux import FLUX # Flux class network
from util import pearsonr
from DatasetFlux import MyDataset



# hyper parameters
LEARN_RATE = 0.008  
# EPOCH = 100
LAMB_BA = 1
LAMB_NG = 1 
LAMB_CELL =  1
LAMB_MOD = 1e-2 


def myLoss(m, c, lamb1 = 0.2, lamb2= 0.2, lamb3 = 0.2, lamb4 = 0.2, geneScale = None, moduleScale = None):    
    
    # balance constrain
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim = 1) 
    
    # non-negative constrain
    error = torch.abs(m) - m  # m is SC * N_module
    total2 = torch.sum(error, dim=1)
    
    
    # sample-wise variation constrain 
    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    #total3 = torch.pow(diff, 0.5)
    if sum(diff > 0) == m.shape[0]: # solve Nan after several iteraions
        total3 = torch.pow(diff, 0.5)
    else:
        print('find 0 in loss three.')
        total3 = diff
    
    
    # module-wise variation constrain
    if lamb4 > 0 :
        corr = torch.FloatTensor(np.ones(m.shape[0]))
        for i in range(m.shape[0]):
            corr[i] = pearsonr(m[i, :], moduleScale[i, :])
        corr = torch.abs(corr)
        penal_m_var = torch.FloatTensor(np.ones(m.shape[0])) - corr
        total4 = penal_m_var
    else:
        total4 = torch.FloatTensor(np.zeros(m.shape[0]))
            
    # loss
    loss1 = torch.sum(lamb1 * total1)
    loss2 = torch.sum(lamb2 * total2)
    loss3 = torch.sum(lamb3 * total3)
    loss4 = torch.sum(lamb4 * total4)
    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss1, loss2, loss3, loss4


def main(args):
    
    # set arguments
    data_path = args.data_dir
    input_path = args.input_dir
    res_dir = args.res_dir
    test_file = args.test_file
    moduleGene_file = args.moduleGene_file
    cm_file = args.stoichiometry_matrix
    sc_imputation = args.sc_imputation
    cName_file = args.cName_file
    gradName = args.output_gradient_file
    EPOCH = args.train_epoch
    #fileName = args.output_flux_file
    #balanceName = args.output_balance_file
    
    if EPOCH <= 0:
        raise NameError('EPOCH must greater than 1!')

    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # read data
    print("Starting load data...")
    geneExpr = pd.read_csv(
                input_path + '/' + test_file,
                index_col=0)
    geneExpr = geneExpr.T
    geneExpr = geneExpr * 1.0
    if sc_imputation == True:
        magic_operator = magic.MAGIC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geneExpr = magic_operator.fit_transform(geneExpr)
    if geneExpr.max().max() > 50:
        geneExpr = (geneExpr + 1).apply(np.log2)  
    geneExprSum = geneExpr.sum(axis=1)
    stand = geneExprSum.mean()
    geneExprScale = geneExprSum / stand
    geneExprScale = torch.FloatTensor(geneExprScale.values).to(device)
    
    BATCH_SIZE = geneExpr.shape[0]
    
    moduleGene = pd.read_csv(
                data_path + '/' + moduleGene_file,
                sep=',',
                index_col=0)
    moduleLen = [moduleGene.iloc[i,:].notna().sum() for i in range(moduleGene.shape[0]) ]
    moduleLen = np.array(moduleLen)
    
    # find existing gene
    module_gene_all = []
    for i in range(moduleGene.shape[0]):
        for j in range(moduleGene.shape[1]):
            if pd.isna(moduleGene.iloc[i,j]) == False:
                module_gene_all.append(moduleGene.iloc[i,j])
    module_gene_all = set(module_gene_all)
    data_gene_all = set(geneExpr.columns)
    gene_overlap = list(data_gene_all.intersection(module_gene_all))   # fix
    gene_overlap.sort()

    cmMat = pd.read_csv(
            data_path + '/' + cm_file,
            sep=',',
            header=None)
    cmMat = cmMat.values
    cmMat = torch.FloatTensor(cmMat).to(device)
    
    if cName_file != 'noCompoundName':
        print("Load compound name file, the balance output will have compound name.")
        cName = pd.read_csv(
                "./data/" + cName_file,
                sep=',',
                header=0)
        cName = cName.columns
    print("Load data done.")
    
    print("Starting process data...")
    emptyNode = []
    # extract overlap gene
    geneExpr = geneExpr[gene_overlap] 
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
        geneExprDf = geneExprDf.append(temp, ignore_index = True, sort=False)
    geneExprDf.index = geneExprDf['Module_Gene']
    geneExprDf.drop('Module_Gene', axis = 'columns', inplace = True)
    X = geneExprDf.values.T
    X = torch.FloatTensor(X).to(device)
    
    #prepare data for constraint of module variation based on gene
    df = geneExprDf
    df.index = [i.split('_')[0] for i in df.index]
    df.index = df.index.astype(int)   # mush change type to ensure correct order, T column name order change!
    #module_scale = df.groupby(df.index).sum(axis=1).T   # pandas version update
    module_scale = df.groupby(df.index).sum().T  
    module_scale = torch.FloatTensor(module_scale.values/ moduleLen) 
    print("Process data done.")

    
    
# =============================================================================
    #NN
    torch.manual_seed(16)
    net = FLUX(X, n_modules, f_in = n_genes, f_out = 1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = LEARN_RATE)

    #Dataloader
    dataloader_params = {'batch_size': BATCH_SIZE,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}

    dataSet = MyDataset(X, geneExprScale, module_scale)
    train_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                               **dataloader_params)
    
# =============================================================================

  
    
# =============================================================================
    print("Starting train neural network...")
    start = time.time()  
#   training
    loss_v = []
    loss_v1 = []
    loss_v2 = []
    loss_v3 = []
    loss_v4 = []
    net.train()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    lossName = "./output/lossValue_" + timestr + ".txt"
    file_loss = open(lossName, "a")
    for epoch in tqdm(range(EPOCH)):
        loss, loss1, loss2, loss3, loss4 = 0,0,0,0,0
        
        for i, (X, X_scale, m_scale) in enumerate(train_loader):

            X_batch = Variable(X.float().to(device))
            X_scale_batch = Variable(X_scale.float().to(device))
            m_scale_batch = Variable(m_scale.float().to(device))
            
            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            loss_batch, loss1_batch, loss2_batch, loss3_batch, loss4_batch  = myLoss(out_m_batch, out_c_batch, 
                                                                                     lamb1 = LAMB_BA, lamb2 = LAMB_NG, lamb3 = LAMB_CELL, lamb4 = LAMB_MOD, 
                                                                                     geneScale = X_scale_batch, moduleScale = m_scale_batch)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
            loss += loss_batch.cpu().data.numpy()
            loss1 += loss1_batch.cpu().data.numpy()
            loss2 += loss2_batch.cpu().data.numpy()
            loss3 += loss3_batch.cpu().data.numpy()
            loss4 += loss4_batch.cpu().data.numpy()
            
        #print('epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f, loss4: %.8f, loss: %.8f' % (epoch+1, loss1, loss2, loss3, loss4, loss))
        file_loss.write('epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f, loss4: %.8f, loss: %.8f. \n' % (epoch+1, loss1, loss2, loss3, loss4, loss))
        
        loss_v.append(loss)
        loss_v1.append(loss1)
        loss_v2.append(loss2)
        loss_v3.append(loss3)
        loss_v4.append(loss4)
        
# =============================================================================
    end = time.time()
    print("Training time: ", end - start) 
    
    file_loss.close()
    plt.plot(loss_v, '--')
    plt.plot(loss_v1)
    plt.plot(loss_v2)
    plt.plot(loss_v3)
    plt.plot(loss_v4)
    plt.legend(['total', 'balance', 'negative', 'cellVar', 'moduleVar']);
    imgName = './' + res_dir + '/loss_' + timestr + ".png"
    plt.savefig(imgName)
    timeName =  './' + res_dir + '/time_' + timestr + ".txt"
    f = open(timeName, "a")
    runTimeStr = str(end - start)
    f.write(runTimeStr)
    f.close()  
    
    #save model
    model_name = './' + res_dir + '/model_' + timestr + '.pkl'
    torch.save(net, model_name)
    print("Trained model saved.")
    #load model
#    net1 = torch.load('./output/model_GSE103322_m171_top20_20201012-172505.pkl')

# for name, param in net.named_parameters():
#     print(name, '___', param)
# print(net.named_parameters())
# net.state_dict()   

    # 3: gradient by refitting
    print("Start to calculate gradient...")
    # =============================================================================
    n_common = geneExpr.shape[1]
    #    Dataloader
    dataloader_params = {'batch_size': 1,
                             'shuffle': False,
                             'num_workers': 0,
                             'pin_memory': False}
    #    test_loader = torch.utils.data.DataLoader(
    #                                                torch.utils.data.TensorDataset(X),
    #                                                **dataloader_params)
    dataSet = MyDataset(X, geneExprScale, module_scale)
    test_loader = torch.utils.data.DataLoader(dataset=dataSet,
                              **dataloader_params)
    
        
    #   training 
    #        for i, item in enumerate(train_loader): # 162
    df_grad_cell_all = np.zeros((n_cells, n_common, n_modules), dtype='f')
    for i, (X, X_scale, m_scale) in tqdm(enumerate(test_loader)):
    #            X_batch, scale_batch , b_scale_batch= Variable(next(training_loader_iter)[0])  
    #            X_batch = Variable(item[0])
                X_batch = Variable(X.float().to(device))
                X_scale_batch = Variable(X_scale.float().to(device))
                m_scale_batch = Variable(m_scale.float().to(device))
    #            print(X_batch.shape)
    #            print(X_scale_batch.shape)
                
                out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat)
                loss_batch, loss1_batch, loss2_batch, loss3_batch, loss4_batch  = myLoss(out_m_batch, out_c_batch, 
                                                                                         lamb1 = LAMB_BA, lamb2 = LAMB_NG, lamb3 = LAMB_CELL, lamb4 = LAMB_MOD, 
                                                                                         geneScale = X_scale_batch, moduleScale = m_scale_batch)
    #            print(loss_batch)
                # back propagation
                optimizer.zero_grad()
                loss_batch.backward()
                # print("Model's state_dict:")
                # for param_tensor in net.state_dict():
                #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
                #print(net.m_encoder[0][0].weight.grad.numpy())
                #print(net.m_encoder[0][0].weight.grad.numpy().sum())
    
                #------------------------------
                id_weight = 0
                id_module = 0
                countK = 1
                multi_tmp = torch.randn(5)
                #df_weight = np.zeros((n_common, n_modules), dtype='f')
                df_grad_cell_i = np.zeros((n_common, n_modules), dtype='f')
                box1 = torch.randn(5)
                box2 = torch.randn(5)
                #box3 = torch.randn(5)
                for name, param in net.named_parameters():
                    id_weight += 1
                    # print(name, '__\n', )
                    # print(param.data.shape)
                    
                    if countK == 1:
                        box1 = param
                    if countK == 2:
                        box2 = param
                    if countK == 3:
                        #box3 = param
                        countK = 0
                        
                    ##print(id_weight, ',,,,', countK)
                    
                    if id_weight % 3 == 0:
                        # print(box1.shape)
                        # print(box2.shape)
                        # print(box3.shape)
                        # count_tmp += 1
                        # print(count_tmp)
                        ###multi_tmp = torch.matmul(torch.t(box1), torch.t(box2))
                        multi_tmp = torch.matmul(torch.t(box1.grad), torch.t(box2.grad))  #use gradient instead of weight!
                        # multi_tmp = multi_tmp.view(-1)
                        # print(multi_tmp.shape)
                        para_gene = multi_tmp.view(-1).detach().numpy()
                        ##print(para_gene.shape)
                        
                        genes = moduleGene.iloc[id_module,:].values.astype(str)
                        genes = [g for g in genes if g != 'nan']
                        temp = geneExpr.copy()
                        temp.loc[:, [g for g in gene_names if g not in genes]] = 0
                        sel_gene_col = temp.sum(axis = 0)
                        unit_sel_gene = sel_gene_col / sel_gene_col
                            
                        df_grad_cell_i[:, id_module] = np.multiply(para_gene, unit_sel_gene)
                        id_module += 1
                    
                    countK += 1
                #-------------------------------
                df_grad_cell_all[i, :, :] = df_grad_cell_i
                
                # if i == 0:  #debug
                #     break
                
                optimizer.step()
    print("Gradient calculation is done.")
    mySaveGene = geneExpr.columns
    mySaveModule = moduleGene.index
    mySaveCell = geneExpr.index
    
    # Saving the objects:
    if gradName == 'NULL':
        # user do not define file name of balance
        gradName = './' + res_dir + '/calcGrad_' + timestr +'.pkl'
  
    with open(gradName, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([df_grad_cell_all, mySaveGene, mySaveModule, mySaveCell], f) 

    # # Getting back the objects:
    # with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
    #     obj0, obj1, obj2 = pickle.load(f)
    
    
    print("scFEA gradient calculation job finished. Check result in pkl extension file.")
    
    return



def parse_arguments(parser):


    parser.add_argument('--data_dir', type=str, default='data', metavar='<data_directory>',
                        help='The data directory for scFEA model files.')
    parser.add_argument('--input_dir', type=str, default='data', metavar='<input_directory>',
                        help='The data directory for single cell input data.')
    parser.add_argument('--res_dir', type=str, default='output', metavar='<data_directory>',
                        help='The data directory for result [output]. The output of scFEA includes two matrices, predicted metabolic flux and metabolites stress at single cell resolution.')
    parser.add_argument('--test_file', type=str, default='Melissa_full.csv', 
                        help='The test SC file [input]. The input of scFEA is a single cell profile matrix, where row is gene and column is cell. Example datasets are provided in /data/ folder. The input can be raw counts or normalised counts. The logarithm would be performed if value larger than 30.')
    parser.add_argument('--moduleGene_file', type=str, default='module_gene_m168.csv', 
                        help='The table contains genes for each module. We provide human and mouse two models in scFEA. For human model, please use module_gene_m168.csv which is default.  All candidate moduleGene files are provided in /data/ folder.')
    parser.add_argument('--stoichiometry_matrix', type=str, default='cmMat_c70_m168.csv', 
                        help='The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_171.csv which is default. All candidate stoichiometry matrices are provided in /data/ folder.')
    parser.add_argument('--cName_file', type=str, default='cName_c70_m168.csv',
                        help='The name of compounds. The table contains two rows. First row is compounds name and second row is corresponding id.')
    parser.add_argument('--sc_imputation', type=eval, default='False', choices=[True, False],
                        help='Whether perform imputation for SC dataset (recommend set to <True> for 10x data).')
    parser.add_argument('--output_gradient_file', type=str, default='NULL', 
                        help='User defined calculated gradient file name. (please use .pkl extension)')
    parser.add_argument('--train_epoch', type=int, default=100, nargs='?',
                        help='User defined EPOCH (training iteration).')
    # parser.add_argument('--output_flux_file', type=str, default='NULL', 
    #                     help='User defined predicted flux file name.')
    # parser.add_argument('--output_balance_file', type=str, default='NULL', 
    #                     help='User defined predicted balance file name.')
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='scFEA-gradient, calculate gradient by refitting the model of each cell. Please visit http://scflux.org/ for more instruction.')
    args = parse_arguments(parser)
    main(args)
