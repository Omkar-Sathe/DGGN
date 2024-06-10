from node2vec import Node2Vec
import os
import networkx as nx
import numpy as np

import random


geneidx = [] 
filea = open("data/gidx.txt")
node_gene = filea.readlines()
for id in node_gene:
    node_1 = id.split()[0]
    geneidx.append(node_1)

doidx = []
filea = open("data/didx.txt")
node_disease = filea.readlines()
for id in node_disease:
    node_1 = id.split()[0]
    doidx.append(node_1)


filea = open("data/balanced data/gd/X_train.txt")
x_train = filea.readlines()
fileb = open("data/balanced data/gd/y_train.txt")
y_train = fileb.readlines()

filea = open("data/balanced data/gd/X_test.txt")
X_test = filea.readlines()
fileb = open("data/balanced data/gd/y_test.txt")
y_test = fileb.readlines()
# train set
train = []
for i in range(len(x_train)):
        tup1 = (x_train[i].split()[0].split(",")[0],)
        tup2 = (x_train[i].split()[0].split(",")[1],)
        tup3 = (y_train[i].split()[0],)
        tup4 = tup1 + tup2 + tup3
        train.append(tup4) 
# test set      
test = []
for i in range(len(X_test)):
        tup1 = (X_test[i].split()[0].split(",")[0],)
        tup2 = (X_test[i].split()[0].split(",")[1],)
        tup3 = (y_test[i].split()[0],)
        tup4 = tup1 + tup2 + tup3
        test.append(tup4) 
print("appeded.....")
if __name__ == '__main__':
    print("entered the if statement.......")
    gmd_graph = nx.read_weighted_edgelist('data/gg_dd_edges.edgelist',delimiter="\t")  
    print(gmd_graph)
    print("Before:",gmd_graph.number_of_nodes(),gmd_graph.number_of_edges())
    for edge in list(gmd_graph.edges):
        if gmd_graph.get_edge_data(edge[0],edge[1])['weight']<=0.0:
            gmd_graph.remove_edge(edge[0],edge[1])
    print("After:",gmd_graph.number_of_nodes(),gmd_graph.number_of_edges())

    node2vec = Node2Vec(gmd_graph, dimensions=64, walk_length=10, num_walks=100, workers=4) 
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("after fitting.........")
    gd_g_vec={} 
    for gid in geneidx:
        gd_g_vec[gid]=model.wv.get_vector(gid)  
    print("disease vector obtained........")
    gd_d_vec={}                 
    for dis in doidx:
        gd_d_vec[dis]=model.wv.get_vector(dis)  


    X_train_gd =[]          
    for gd in train:                    
        gd_vec = np.concatenate((gd_g_vec[gd[0]],gd_d_vec[gd[1]])) 
        X_train_gd.append(gd_vec)
        
    X_train_gd = np.array(X_train_gd)
    X_test_gd =[]
    for gd in test:
        gd_vec = np.concatenate((gd_g_vec[gd[0]],gd_d_vec[gd[1]]))
        X_test_gd.append(gd_vec)
        
    X_test_gd = np.array(X_test_gd) 
    print(X_train_gd.shape, X_test_gd.shape)
    np.save('X_train_gg_dd_node2vec.npy',X_train_gd) 
    np.save('X_test_gg_dd_node2vec.npy',X_test_gd)


    print("save successful！！！")