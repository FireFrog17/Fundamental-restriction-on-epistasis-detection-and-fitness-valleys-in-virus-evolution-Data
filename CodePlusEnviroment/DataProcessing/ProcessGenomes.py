
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sm


class Matrix:

    def __init__(self, FilePath, iterator):
        with open(FilePath) as genoms_file:
            genoms = genoms_file.read()
        NASdict = {}
        NamesAndSequence = genoms.split('>')

        for NameAndSeq in NamesAndSequence[1 : iterator * 1000+1]:
            NASlist = NameAndSeq.split('\n')
            NASdict[NASlist[0]] = list("".join(NASlist[1:]))
            #print(NASdict)

        self.df = pd.DataFrame.from_dict(NASdict, orient='index')

        self.origin = self.df.copy()

        #print(self.origin)

    def columns_cleaning(self, SiteGapsThreshold, SiteMonomorphThreshold):
        DropList = []
        '''
        Deletes column(site) if percentage of gaps is over SiteGapsThreshold
        '''
        for i in self.df.columns:
            if '-' in self.df.loc[:, i].value_counts():
                if self.df.loc[:, i].value_counts()['-'] / self.df.shape[0] >= SiteGapsThreshold:
                    DropList.append(i)
                  #  print("bad spot cleaned")
        '''
        Deletes column(site) if percentage of consensual letters is more then SiteMonomorphThreshold
        '''
        #SiteMonomorphThreshold = 1 - SiteMonomorphThreshold
        for i in self.df.columns:
            if len(self.df.loc[:, i].value_counts()) >= 2:
                if self.df.loc[:, i].value_counts().max() / self.df.shape[0] >= SiteMonomorphThreshold:
                    DropList.append(i)
                    #print("mono spot cleaned")
            elif len(self.df.loc[:, i].value_counts()) == 1:
                DropList.append(i)
               # print("mono spot cleaned")
        DropList = list(set(DropList))
        #print(DropList)
        #print("spot cleaning finished")
        self.df.drop(DropList, inplace=True, axis=1)


    def index_cleaning(self, SeqGapsThreshold):
        """
        Deletes sequence if percentage of gaps in it is over SeqGapsThreshold
        """
        DropList = []
        for i in self.df.index:
            if '-' in self.df.loc[i, :].value_counts():
                if self.df.loc[i, :].value_counts()['-'] / self.df.shape[1] >= SeqGapsThreshold:
                    DropList.append(i)
                    #print("seq cleaned!")
        #print(DropList)
        #print("seq cleaning finished")
        self.df.drop(DropList, inplace=True)

    def binarization(self):
        """
        Binarization,0 - consensual variant, 1 - anti consensual variant
        """
        #print(self.df)
        for i in self.df.columns:
            self.df.loc[self.df[i] == self.df.loc[:, i].value_counts().idxmax(), i] = 0
            self.df.loc[self.df[i] != self.df.loc[:, i].value_counts().idxmax(), i] = 1

    def onrefbin(self):
        for i in self.df.columns:
            self.df.loc[self.df[i] == self.df.loc[:,i][0], i] = 0
            self.df.loc[self.df[i] != self.df.loc[:,i][0], i] = 1



newMatrix = Matrix("YourAlignedSequences.fasta", 20)
newMatrix.index_cleaning(0.7)
newMatrix.columns_cleaning(0.5, 0.8)
newMatrix.onrefbin()
#print(newMatrix.df.shape)
#print(newMatrix.df.columns)
#print(newMatrix.df.values)
np.save("YourDataNameVal.npy", newMatrix.df.values)
np.save("YourDataNameCol.npy", newMatrix.df.columns)
arrayFull = np.transpose(newMatrix.df.values)
plt.hist(np.sum(arrayFull,axis=0),round(np.sqrt(arrayFull.shape[1])))
plt.title("Mutation hist")
plt.xlabel("Number of mutations")
plt.savefig("YourDataPath"+".png")