# -*- coding: utf-8 -*-
"""

@author: wnchang@iu.edu
"""

# system lib
import argparse
import time
import warnings

# tools
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import magic
from tqdm import tqdm

# scFEA lib
from ClassFlux import FLUX # Flux class network
from util import pearsonr
from DatasetFlux import MyDataset


# hyper parameters
LEARN_RATE = 0.008  
EPOCH = 100
LAMB_BA = 1
LAMB_NG = 1 
LAMB_CELL =  1
LAMB_MOD = 1e-2 


def myLoss(m, c, lamb1 = 0.2, lamb2= 0.2, lamb3 = 0.2, lamb4 = 0.2, geneScale = None, moduleScale = None):    
    
    # balance constrain
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim = 1) 
    
    # non-negative constrain
    error = torch.abs(m) - m
    total2 = torch.sum(error, dim=1)
    
    
    # sample-wise variation constrain 
    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    #total3 = torch.pow(diff, 0.5)
    if sum(diff > 0) == m.shape[0]: # solve Nan after several iteraions
        total3 = torch.pow(diff, 0.5)
    else:
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
    gene_overlap = data_gene_all.intersection(module_gene_all)
    
    cmMat = pd.read_csv(
            data_path + '/' + cm_file,
            sep=',',
            header=None)
    cmMat = cmMat.values
    cmMat = torch.FloatTensor(cmMat).to(device)
    
    if cName_file != 'noCompoundName':
        print("Load compound name file, the balance output will have compound name.")
        cName = pd.read_csv(
                "./data/" + cName_file + ".csv",
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

    
#    Dataloader
    dataloader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}

    dataSet = MyDataset(X, geneExprScale, module_scale)
    test_loader = torch.utils.data.DataLoader(dataset=dataSet,
                          **dataloader_params)
   
    #testing
    fluxStatuTest = np.zeros((n_cells, n_modules), dtype='f') #float32
    balanceStatus = np.zeros((n_cells, n_comps), dtype='f')
    net.eval()
    for epoch in range(1):
        loss, loss1, loss2 = 0,0,0
        
        for i, (X, X_scale, _) in enumerate(test_loader):

            X_batch = Variable(X.float().to(device))
            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            
            # save data
            fluxStatuTest[i, :] = out_m_batch.detach().numpy()
            balanceStatus[i, :] = out_c_batch.detach().numpy()
            
                   
    
    # save to file
    fileName = "./" + res_dir + test_file[-len(test_file):-4] + "/module" + str(n_modules) + "_cell" + str(n_cells) + "_batch" + str(BATCH_SIZE) + \
                "_LR" + str(LEARN_RATE) + "_epoch" + str(EPOCH) + "_SCimpute_" + str(sc_imputation)[0] + \
                "_lambBal" + str(LAMB_BA) + "_lambSca" + str(LAMB_NG) + "_lambCellCor" + str(LAMB_CELL) + "_lambModCor_1e-2" + \
                '_' + timestr + ".csv"
    setF = pd.DataFrame(fluxStatuTest)
    setF.columns = moduleGene.index
    setF.index = geneExpr.index.tolist()
    setF.to_csv(fileName)
    
    setB = pd.DataFrame(balanceStatus)
    setB.rename(columns = lambda x: x + 1)
    setB.index = setF.index
    if cName_file != 'noCompoundName':
        setB.columns = cName
    balanceName = "./output/balance_" + timestr + ".csv"
    setB.to_csv(balanceName)
    

    print("scFEA job finished. Check result in the desired output folder.")
    
    
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
    parser.add_argument('--moduleGene_file', type=str, default='module_gene_m171_vDec2020.csv', 
                        help='The table contains genes for each module. We provide human and mouse two models in scFEA. For human model, please use module_gene_m171_vDec2020.csv which is default. For mouse model, please use module_gene_mouse_m162.csv. All candidate moduleGene files are provided in /data/ folder.')
    parser.add_argument('--stoichiometry_matrix', type=str, default='cmMat_m171.csv', 
                        help='The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_171.csv which is default. For mouse model, please use cmMat_mouse_c66_m162.csv. All candidate stoichiometry matrices are provided in /data/ folder.')
    parser.add_argument('--cName_file', type=str, default='noCompoundName',
                        help='The name of compounds. The table contains two rows. First row is compounds name and second row is corresponding id.')
    parser.add_argument('--sc_imputation', type=eval, default='False', choices=[True, False],
                        help='Whether perform imputation for SC dataset (recommend set to <True> for 10x data).')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='scFEA: A graph neural network model to estimate cell-wise metabolic flux using single cell RNA-seq data')
    args = parse_arguments(parser)
    main(args)