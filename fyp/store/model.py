# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:56:03 2019

@author: kafee
"""

import pandas as pd
from datetime import datetime
import collections
import numpy as np
import networkx as nx
from sklearn.neighbors.kde import KernelDensity
import math
import itertools
from sklearn import metrics
import copy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle


class Model:
    
    
    
    def __init__(self,filename):
        self.graph=self.SuspiciousRG(filename=filename)
        self.cliques=self.Cliques(self.graph.suspicious_graph)
        self.final_groups=self.Features(self.graph.raw_data_df,self.graph.grouped_df,self.cliques)
        self.testLabel=self.final_groups.testLabel
        self.predictions=self.final_groups.predictions
        self.data_for_model=self.final_groups.data_for_model
    
    def accuracy_score(self):
        return metrics.accuracy_score(self.testLabel,self.predictions)
    
    def confusion_matrix(self):
        return metrics.confusion_matrix(self.testLabel,self.predictions)
    
    def predict(self,R_id,p_id,rating,label,date):
        raw_data_frame=self.graph.raw_data_df
        Candidate_groups=self.cliques.candidate_groups
        filter_data_frame=raw_data_frame[raw_data_frame['asin']==p_id]
        filter_data_frame=filter_data_frame.append({'reviewerID' : R_id , 'asin' : p_id , 'label' : label , 'overall' : rating , 'reviewTime' : date} , ignore_index=True)
        filter_data_frame=filter_data_frame.drop_duplicates() 
        graph=self.SuspiciousRG(filter_data_frame=filter_data_frame)
        graph=graph.suspicious_graph
        graph=graph[(graph['reviewer_1']==R_id) | (graph['reviewer_2']==R_id)]
        rev=0
        length=0
        
        for index,row in graph.iterrows():
            if row['reviewer_1']==R_id:
               for i in Candidate_groups:
                   if row['reviewer_2'] in Candidate_groups[i]:
                       if len(Candidate_groups[i])>=self.cliques.k and len(Candidate_groups[i])>length:
                           length=len(Candidate_groups[i])
                           rev=row['reviewer_2']
                           
                           
            elif row['reviewer_2']==R_id:
                for i in Candidate_groups:
                   if row['reviewer_1'] in Candidate_groups[i]:
                       if len(Candidate_groups[i])>=self.cliques.k and len(Candidate_groups[i])>length:
                           length=len(Candidate_groups[i])
                           rev=row['reviewer_1']
        
        sort_by_score=self.final_groups.final_df
        suspicious_score = 0
        Group_Rating_Dev = 0
        avg_rating_dev = 0
        burst_rt = 0
        Group_Size_Rt = 0
        review_tght = 0
        Group_Support = 0
        Group_size = 0
        Group_num=0
        Group_reviwers=0
        Group_Content_Similarity=0
        
        flag=0
        for index, row in sort_by_score.iterrows():
            for index_list, val in enumerate(row['Groups']):
                if rev==val:
                    if flag==1:
                        break
                    flag=1
                    Group_Rating_Dev = row['group_deviation']
                    avg_rating_dev = row['avg_rating_deviation']
                    burst_rt = row['burst_ratio']
                    Group_size = row['group_size']
                    Group_Size_Rt = row['group_size_ratio']
                    review_tght = row['review_tightness']
                    Group_Support = row['group_support']
                    Group_Content_Similarity=row['group_content_similarity']
                    suspicious_score=row['suspicious_score']
                    Group_num=index
                    Group_reviwers=len(row['Groups'])
        string=""
        if flag==1:
            string+="Total Number of Groups = "+ str(len(sort_by_score.index)) +"\nGroup Spam belongs to group "+ str(Group_num) + " with Group members "+ str(Group_reviwers) +" having suspicious score "+ str(suspicious_score) +" with the following Indicators that makes this reviewer spam" + "\nGroup Deviation "+str(round(Group_Rating_Dev,2))+"\nAverage Rating Deviation "+str(round(avg_rating_dev,2))+"\nBurst Ratio "+str(round(burst_rt,2))+"\nGroup Size "+str(round(Group_size,2))+"\nGroup Size Ratio "+str(round(Group_Size_Rt,2))+"\nReview Tightness "+str(round(review_tght,2))+"\nGroup Support Count "+str(round(Group_Support,2))+"\nGroup Content Similarity "+str(round(Group_Content_Similarity,2))
            string+="Spam"
        
        else:
            string+="Not Spam"
        
        return string
        #return sort_by_score,filter_data_frame
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName,"wb")
        pickle.dump(self,f)
        f.close()
        
    def load(fileName):
        """Return a thing loaded from a file."""
        f = open(fileName,"rb")
        obj = pickle.load(f)
        f.close()
        return obj
    load = staticmethod(load)
           
        
    class SuspiciousRG:
        
        helo=0
        graph_df=pd.DataFrame(data=None, columns=['reviewer_1', 'reviewer_2', 'suspicious_score','commonproducts'])
        
        def __init__(self,filename=None,filter_data_frame=None):
            self.suspicious_graph,self.raw_data_df,self.grouped_df=self.Graph(datafile=filename,filter_data_frame=filter_data_frame)
        
        def lambdaFunction(self,row_reviewer1,row_reviewer2):
            val=self.RSG(row_reviewer1['ordinal'],row_reviewer1['overall'],row_reviewer2['ordinal'],row_reviewer2['overall'])
            if val == 1:
                return 1
            return 0
        
        def RSG(self,time1,rating1,time2,rating2):
            alpha=10
            if (abs(time1-time2))>alpha or (abs(rating1-rating2))>=2:
                return 0
            
            else:
                return 1
        def calculate1(self,df1,df):
            global graph_df
            val=self.lambdaFunction(df,df1)
            if val == 1:
                length=len(self.graph_df[(self.graph_df['reviewer_1']==df['reviewerID']) & (self.graph_df['reviewer_2']==df1['reviewerID'])])
                if(length==0):
                    self.graph_df=self.graph_df.append({'reviewer_1' : df['reviewerID'] , 'reviewer_2' : df1['reviewerID'] , 'suspicious_score' : 1 , 'commonproducts' : 1} ,ignore_index=True)
                else:
                    self.graph_df.loc[(self.graph_df['reviewer_1'] == df['reviewerID']) & (self.graph_df['reviewer_2'] == df1['reviewerID']), 'commonproducts'] += 1
        
        def calculate(self,df):
            global helo
            self.helo=self.helo.loc[(self.helo['reviewerID']!=df['reviewerID'])|(self.helo['overall']!=df['overall'])]
            self.helo.apply(self.calculate1,df=df,axis=1)
            
        def Graph(self,datafile,filter_data_frame):
            global helo
            if filter_data_frame is None:
                csv_file = datafile
                raw_data_df = pd.read_csv(csv_file, sep="\t")
            else:
                raw_data_df=filter_data_frame
                
            #raw_data_df=raw_data_df.drop_duplicates().reset_index(drop=True)
            reviewers_df = pd.DataFrame(columns= raw_data_df.columns.values)
            for col in  raw_data_df.columns.values:
                if (col == "reviewerID"): continue
                reviewers_df[col] =  raw_data_df.groupby("reviewerID")[col].apply(list)
            date_conv = lambda row: [datetime.strptime(r, '%Y-%m-%d').date().toordinal() for r in row.reviewTime]
            reviewers_df['ordinal'] = reviewers_df.apply(date_conv, axis=1)
            reviewers_df.drop('reviewerID', axis=1, inplace=True)
            
            date_conv1 = lambda row: datetime.strptime(row['reviewTime'], '%Y-%m-%d').date().toordinal()
            raw_data_df['ordinal'] = raw_data_df.apply(date_conv1, axis=1)
            for i, g in raw_data_df.groupby('asin'):
                if(len(g)>1):
                    self.helo=g.reset_index(drop=True).copy()
                    g.apply(self.calculate,axis=1)
            return self.graph_df,raw_data_df,reviewers_df
            
        
        
                
   
    class Cliques:
        k=3
        def __init__(self,graphdf):
            self.candidate_groups=self.cliques(graphdf)
        
        def Intersection(self,lst1, lst2): 
            return list(set(lst1).intersection(lst2))  
    
        def cliques(self,graphdf):
            
            groups=dict()
            cliques=dict()
            common_products=collections.defaultdict(dict)
            cliques_products=collections.defaultdict(dict)
            kcliques=[]
            for index, row in graphdf.iterrows():
                reviewer1 = str(row["reviewer_1"])
                reviewer2 = str(row["reviewer_2"])
                product   = str(row["commonproducts"])
                if reviewer1 in groups:
                    groups[reviewer1].append(reviewer2)
                    common_products[reviewer1][reviewer2]=product
                else:
                    groups[reviewer1]=[reviewer2]
                    common_products[reviewer1][reviewer2]=product
            
            for reviewers in groups:
                for reviewer in groups[reviewers]:
                    for keys in groups:
                        if keys==reviewers:
                            continue
                        elif keys==reviewer:
                            commonid=self.Intersection(groups[reviewers],groups[keys])
                            if not commonid:
                                continue
                            else:
                                if reviewers in cliques:
                                    commonid.append(reviewer)
                                    for i in commonid:
                                        if i in cliques[reviewers]:
                                            continue
                                        else:
                                            cliques[reviewers].append(i)
                                else:
                                    cliques[reviewers]=commonid
                                    cliques[reviewers].append(reviewer)
            
            for r1,v1 in common_products.items():
                for r2,v2 in cliques.items():
                    if r1==r2:
                        kcliques.append(len(cliques[r2])+1)
                        for value1 in v1 :
                            if value1 in cliques[r2]:
                                cliques_products[r1][value1]=v1[value1]
                    else:
                        continue
            
            new_graph = nx.Graph()
            for source, targets in cliques_products.items():
                for source1,targets1 in cliques_products.items():
                    if source==source1:
                        continue
                    else:
                         commonIntersect=list(set( cliques_products[source].items() ) & set( cliques_products[source1].items() ))
                         if not commonIntersect:
                             accumulatedProduct=0
                             new_graph.add_edge(str(source), str(source1),weight=int(accumulatedProduct))
                         else:
                             accumulatedProduct=0
                             for x in commonIntersect:
                                 accumulatedProduct=accumulatedProduct + int(x[1])
                             new_graph.add_edge(str(source), str(source1),weight=int(accumulatedProduct))
                                     
            adjacency_matrix = nx.adjacency_matrix(new_graph)  
            matrix=adjacency_matrix.todense()              
            
            for i in range(len(kcliques)):
               if kcliques[i]<self.k:
                   kcliques[i]=0
               else:
                   kcliques[i]=1
            
            row,col = np.diag_indices(matrix.shape[0])
            matrix[row,col] = kcliques
            # threshold matrix
            for i in row:
                for j in col:
                    if i==j:
                        continue
                    else:
                        if matrix[i,j]<self.k-1:
                            matrix[i,j]=0
                        else:
                            matrix[i,j]=1
            
            
            #Candidate Groups
            Adjacent_Groups=dict()
            Non_Adjacent_Groups=[]
            NodeList=list(new_graph.nodes())
            for i in row:
                reviwer1=NodeList[i]
                group=[]
                if kcliques[i]==0:
                    continue
                for j in col:
                    if i==j:
                        continue
                            
                    else:
                        if matrix[i,j]==1 and kcliques[i]==1:
                            if reviwer1 in Adjacent_Groups:
                                Adjacent_Groups[reviwer1].append(NodeList[j])
                            else:
                                Adjacent_Groups[reviwer1]=[NodeList[j]]
                        if kcliques[i]==1:
                            group.append(matrix[i,j])
                if all(v == 0 for v in group):
                    Non_Adjacent_Groups.append(reviwer1)
            
            #corresponding cliques
            Adj_grp=Adjacent_Groups.copy()
            for reviewers in Adjacent_Groups:
                for reviewer in Adjacent_Groups[reviewers]:
                    if reviewers not in Adj_grp:
                        continue
                    for keys in Adjacent_Groups:
                        if keys==reviewers:
                            continue
                        elif keys==reviewer:
                            for i in Adjacent_Groups[keys]:
                                if i in Adj_grp[reviewers]:
                                    continue
                                else:
                                    Adj_grp[reviewers].append(i)
                            st='Null'+str(keys)
                            Adj_grp[st] = Adj_grp[keys] 
                            del Adj_grp[keys]
            
            #making groups
            index_Adj=0
            Group_keys=list(Adj_grp.keys())
            Candidate_Groups=dict()
            for i in range(len(Adj_grp)):
                key=Group_keys[i]
                if key not in cliques:
                    continue
                Candidate_Groups[i]=[key]
                for values in cliques[key]:
                    Candidate_Groups[i].append(values)
                for values in Adj_grp[key]:
                    for rev in cliques[values]:
                        if rev in Candidate_Groups[i]:
                            continue
                        else:
                            Candidate_Groups[i].append(rev)
                index_Adj=index_Adj+1                   
            
            for i in range(len(Non_Adjacent_Groups)):
                key=Non_Adjacent_Groups[i]
                Candidate_Groups[index_Adj]=[key]
                for values in cliques[key]:
                    Candidate_Groups[index_Adj].append(values)
                index_Adj=index_Adj+1
            return Candidate_Groups
    
    class Features(object):
        
        average = lambda self, n : sum(n) / len(n)
        average2 = lambda self, n : sum(n) / len(n) if len(n) > 0 else 0
        
        Products=collections.defaultdict(list)
        testLabel=[]
        predictions=[]
        
        
        def __init__(self,raw_data_df,grouped_df,cliques):
            
            self.raw_data_df=raw_data_df
            self.grouped_df=grouped_df
            self.Candidate_Groups=cliques.candidate_groups
            self.final_df,self.data_for_model=self.compute_features()
        
            
        
        
        def kde(self,grouped_df):
            product_ratings_list = []
        
            #Grouping ratings and dates by product ID
            for index, row in grouped_df.iterrows():
                for index_list, val in enumerate(row['asin']):
                    product_ratings_list.append([val, grouped_df.loc[index,'overall'][index_list], grouped_df.loc[index,'reviewTime'][index_list]])
            product_ratings = pd.DataFrame(product_ratings_list, columns=["product_id", "rating", "date"])
            grouped_pr = pd.DataFrame(columns=product_ratings.columns.values)
            grouped_pr["rating"] = product_ratings.groupby("product_id")["rating"].apply(list)
            grouped_pr["date"] = product_ratings.groupby("product_id")["date"].apply(list)
        
            #Deleting product_id row as it's already indexed on product_id
            grouped_pr.drop('product_id', axis=1, inplace=True)
        
        
            #Now grouped_pr contains product_id, [list of ratings], [list of dates]
        
            kde_list = []
            p_id = []
        
            # Calculating kde for all dates in all products
            for index, row in grouped_pr.iterrows():
                p_id.append(index)
                for i in range(len(grouped_pr.loc[index,'date'])):
                    #Changing date to ordinal format as kde accepts numerical values only
                    grouped_pr.loc[index, 'date'][i] = [datetime.strptime(grouped_pr.loc[index,'date'][i], '%Y-%m-%d').date().toordinal()]
                grouped_pr.loc[index, 'date'].sort()
                date = np.array(grouped_pr.loc[index, 'date'])
                #Fitting KDE
                kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(date)
                log_dens = kde.score_samples(date)
                kde_list.append(log_dens)
                
            
            #Converting to dataframe : Each columns is a product_id with list of kdes for each sorted date
            kde_df = (pd.DataFrame(kde_list)).transpose()
            kde_df.columns = p_id
        
            #Computing threshold
            kde_df['max'] = kde_df.max(axis = 1)
            sorted_by_max_of_each_row = pd.DataFrame()
            sorted_by_max_of_each_row = kde_df.sort_values(by='max', ascending=False, inplace=False).reset_index(drop=True)
            length=int(len(sorted_by_max_of_each_row['max'])/2)
            threshold = sorted_by_max_of_each_row['max'][length] #threshold is the top 50th value
            kde_df.drop('max', axis=1, inplace=True)
        
            #Filtering all indices with kde > threshold for each product
            intersection = []
            for product_kde in kde_df:
                temp_df = kde_df[kde_df[product_kde]>threshold][product_kde]
                intersection.append(temp_df.index.tolist())
        
            df_indices = (pd.DataFrame(intersection )).transpose()
            df_indices.columns = p_id
        
        
            grouped_pr['bursts'] = grouped_pr['date']
            for column in df_indices:
                b = []
                b_id = []
                id = 0
                a = pd.to_numeric(df_indices[column].dropna())
                if (len(a) == 0):
                    grouped_pr.loc[column, 'bursts'] = []
                    continue
                ini = a[0]
                for i in range(len(a)):
        
                    if (i is (len(a) - 1)) :
                        b.append([grouped_pr.loc[column,"date"][int(ini)],grouped_pr.loc[column,"date"][int(a[i])]])
                        b_id.append([str(column) + '-' + str(id)])
                        id += 1
        
                    elif ((int(a[i]+1) is not int(a[i+1]))):
                        b.append([grouped_pr.loc[column,"date"][int(ini)],grouped_pr.loc[column,"date"][int(a[i])]])
                        b_id.append([str(column) + '-' + str(id)])
                        id += 1
                        ini = a[i+1]
        
                grouped_pr.loc[column,'bursts'] = b
                grouped_pr.loc[column, 'bursts_ids'] = b_id
            return grouped_pr
        
        def reviewer_bursts(self,grouped_df, grouped_pr):
            #Adding ordinal column
            prods_df = grouped_df[['asin', 'reviewTime']]
            date_conv = lambda row: [datetime.strptime(r, '%Y-%m-%d').date().toordinal() for r in row.reviewTime]
            prods_df['ordinal'] = prods_df.apply(date_conv, axis=1)
            prods_df = prods_df.drop('reviewTime', axis=1)
            prods_df = prods_df.reset_index()
        
            grouped_pr = grouped_pr.drop(['rating', 'date'], axis=1)
            calc_count = lambda row: sum([sum([1 for b in grouped_pr['bursts'][prod] if o_time >= b[0][0] and o_time <= b[1][0]]) \
                                        if grouped_pr['bursts'].get(prod) else 0 \
                                        for prod, o_time in zip(row.asin,row.ordinal)])
        
        
            bursts = lambda row: list(filter(None, ([next((b_id for b, b_id \
                                in zip(grouped_pr['bursts'][prod], grouped_pr['bursts_ids'][prod]) \
                                if o_time >= b[0][0] and o_time <= b[1][0]), None) \
                                if grouped_pr['bursts'].get(prod) else None for prod, o_time in zip(row.asin,row.ordinal)])))
            prods_df['burst_ids'] = prods_df.apply(bursts, axis=1)
            prods_df['burst_count'] = prods_df.apply(calc_count, axis=1)
        
            return prods_df
        
        def burst_ratio(self,prods_df):
            find_burst = lambda row: float(row['burst_count'])/len(row.ordinal)
            prods_df['burst_ratio'] = prods_df.apply(find_burst, axis=1)
            prods_df = prods_df.set_index('reviewerID')
            return prods_df
        
        def penalty_function(self,Rg,Pg):
            Penalty_function=1/(1+math.exp(-(Rg+Pg-3)))
            return Penalty_function
        
        def group_size(self,Rg,Group_Size):
            
            size=1/(1+math.exp(-(Rg-3)))
            Group_Size.append(size)
     
        def avg_rating_deviation(self,grouped_df,Candidate_Groups,final_df):
            
            no_of_reviews=0
            total_num_reviews=[]
            average_rating_product = collections.defaultdict(list)
            average_rating_product_group = collections.defaultdict(list)
            Review_tightness=[]
            
            for i in Candidate_Groups:
                for index, row in grouped_df.iterrows():
                    if index in Candidate_Groups[i]:
                        for index_list, val in enumerate(row['asin']):
                            average_rating_product[val].append(grouped_df.loc[index,'overall'][index_list])
                            #Review_Tightness
                            no_of_reviews=no_of_reviews+1
                            self.Products[i].append(val)
                total_num_reviews.append(no_of_reviews)
                no_of_reviews=0
                #Review_Tightness
            
            average_rating_product_group=copy.deepcopy(average_rating_product)
            for key in average_rating_product:
                average_rating_product[key] = sum(average_rating_product[key])/len(average_rating_product[key])
        
            a = []
            Avg_Dev=[]
            Group_Deviation=[]
            a1=[]
            Group_Size=[]
            for i in Candidate_Groups:
                #Group Size
                Rg=len(Candidate_Groups[i])
                self.group_size(Rg,Group_Size)
                #Group Size
                
                #Review_Tightness
                self.review_tightness(Review_tightness,i,total_num_reviews)
                #Review_Tightness
                
                for index, row in grouped_df.iterrows():
                    if index in Candidate_Groups[i]:
                        for index_list, val in enumerate(row['overall']):
                            a.append(abs((val - average_rating_product[grouped_df.loc[index, 'asin'][index_list]])) / 4)
                            
                            #Group Deviation
                            variance=np.var(average_rating_product_group[grouped_df.loc[index, 'asin'][index_list]])
                            a1.append(variance)
                Rg=Candidate_Groups[i]
                Pg=self.Products[i]
                grd=2*(1-(1/(1+math.exp(-self.average(a1)))))*self.penalty_function(len(set(Rg)),len(set(Pg)))
                Group_Deviation.append(grd)
                a1=[]
                #Group Deviation
                
                Avg_Dev.append(self.average(a))
                a = []
            final_df['avg_rating_deviation']=Avg_Dev
            final_df['group_deviation']=Group_Deviation
            final_df['group_size']=Group_Size
            final_df['review_tightness']=Review_tightness
            return final_df
        
        def review_tightness(self,Review_tightness,i,total_num_reviews):
            Rg=self.Candidate_Groups[i]
            Pg=self.Products[i]
            Vg=total_num_reviews[i]
            cartesian_product=len(set(itertools.product(Rg,Pg)))
            RT=(Vg/cartesian_product)*self.penalty_function(len(set(Rg)),len(set(Pg)))
            Review_tightness.append(RT)
        
        
        def BST(self,grouped_df,Candidate_Groups):
            
            Group_Burst_Ratio=[]
            for i in Candidate_Groups:
                temp=[]
                for index, row in grouped_df.iterrows():
                    if index in Candidate_Groups[i]:
                        temp.append(row['burst_ratio'])
                Group_Burst_Ratio.append(self.average(temp))
            
            normalized = [float(i)/max(Group_Burst_Ratio) for i in Group_Burst_Ratio]
            return normalized
        
        def Group_Support_Count(self,Candidate_Groups):
            
            Group_Support_Count=dict()
            for i in Candidate_Groups:
                g=self.Products[i]
                Group_Support_Count[i]=((len(set(g))))
            
            a=[k for k in Group_Support_Count.keys() if Group_Support_Count.get(k)==max([n for n in Group_Support_Count.values()])]
            t=Group_Support_Count[a[0]]
            
            list_temp=[]
            
            for i in range(len(Group_Support_Count)):
                list_temp.append(Group_Support_Count[i]/t)
            
            return list_temp
        
        def Group_Size_Ratio(self,raw_data_df,Candidate_Groups):
            
            grouped_df = pd.DataFrame(columns= raw_data_df.columns.values)
            for col in  raw_data_df.columns.values:
                if (col == "asin"): continue
                grouped_df[col] =  raw_data_df.groupby("asin")[col].apply(list)
        
            grouped_df.drop('asin', axis=1, inplace=True)
        
            Products_id=collections.defaultdict(list)
            for index,row in grouped_df.iterrows():
                for index_list, val in enumerate(row['reviewerID']):
                    Products_id[index].append(val)
            
            Group_size_ratio=collections.defaultdict(list)
            for index in Products_id:
                for i in Candidate_Groups:
                    a=[]
                    for val in Candidate_Groups[i]:
                        if val in Products_id[index]:
                            a.append(val)
                    if len(a)==0:
                        continue
                    else:
                        Group_size_ratio[i].append(len(Candidate_Groups[i])/len(a))
                        a=[]
            
            gsr=[]
            for i in Group_size_ratio:
                gsr.append(self.average(Group_size_ratio[i]))
             
            normalized = [float(i)/max(gsr) for i in gsr]
            return normalized
        def Group_Content_Similarity(self,raw_data_df,Candidate_Groups):
            a=collections.defaultdict(list)
            previous = next_ = None
            data1 = data2 = None
            GCS_list=[]
            for i in Candidate_Groups:
                l = len(Candidate_Groups[i])
                for index, obj in enumerate(Candidate_Groups[i]):
                        if index > 0:
                            previous = Candidate_Groups[i][index - 1]
                            data1=raw_data_df[raw_data_df['reviewerID']==previous]
                        if index < (l - 1):
                            next_ = Candidate_Groups[i][index + 1]
                            data2=raw_data_df[raw_data_df['reviewerID']==next_]
                        if data1 is not None and data2 is not None:
                            data=data1.append(data2)
                            data.reset_index(inplace = True)
                            rowiter = data.iterrows()
                            for index_reviewer1, row_reviewer1 in rowiter:
                                nextrowiter = data.iloc[index_reviewer1 + 1:, :].iterrows()
                                for index_reviewer2, row_reviewer2 in nextrowiter:
                                
                                    if row_reviewer1['asin']==row_reviewer2['asin']:
                                        
                                        X =row_reviewer1['reviewText']
                                        Y= row_reviewer2['reviewText']
                                          
                                        # tokenization 
                                        X_list = word_tokenize(X)  
                                        Y_list = word_tokenize(Y) 
                                          
                                        # sw contains the list of stopwords 
                                        sw = stopwords.words('english')  
                                        l1 =[];l2 =[] 
                                          
                                        # remove stop words from string 
                                        X_set = {w for w in X_list if not w in sw}  
                                        Y_set = {w for w in Y_list if not w in sw} 
                                          
                                        # form a set containing keywords of both strings  
                                        rvector = X_set.union(Y_set)  
                                        for w in rvector: 
                                            if w in X_set: l1.append(1) # create a vector 
                                            else: l1.append(0) 
                                            if w in Y_set: l2.append(1) 
                                            else: l2.append(0) 
                                        c = 0
                                          
                                        # cosine formula  
                                        for j in range(len(rvector)): 
                                                c+= l1[j]*l2[j] 
                                        cosine = c / float((sum(l1)*sum(l2))**0.5)
                                        a[row_reviewer1['asin']].append(cosine)
                list1=[]                           
                for k in a:
                    list1.append(self.average(a[k]))
            
                if len(list1)==1:
                    GCS_list.append(list1[0])
                else:
                    GCS_list.append(max(list1))
                a.clear()
            return GCS_list
    
        
        def getPredictions(self,grouped_df,Candidate_Groups):
            grouped_df_copy=grouped_df.copy()
            sp=[]
            for i in Candidate_Groups:
                for r1 in Candidate_Groups[i]:
                    sp.append(r1)
            sp=list(dict.fromkeys(sp))
                    
            for index,row in grouped_df.iterrows():
                if row['reviewerID'] in sp:
                    grouped_df_copy.loc[index,'label']=1
                    self.testLabel.append(row['label'])
                    self.predictions.append(1)
                else :
                    grouped_df_copy.loc[index,'label']=-1
                    self.testLabel.append(row['label'])
                    self.predictions.append(-1)
            return grouped_df_copy
            
            
        
        
        def compute_features(self):
            
            for i in self.Candidate_Groups:
                self.Candidate_Groups[i] = list(map(int, self.Candidate_Groups[i]))
            
            final_df=pd.DataFrame(columns=['Groups'])
            for i in self.Candidate_Groups:
                final_df.loc[i,'Groups']=self.Candidate_Groups[i]
            
            
            
            final_df = self.avg_rating_deviation(self.grouped_df,self.Candidate_Groups,final_df)
            
            grouped_pr = self.kde(self.grouped_df)
            prods_df = self.reviewer_bursts(self.grouped_df, grouped_pr)
        
            prods_df = self.burst_ratio(prods_df)
            self.grouped_df['burst_ratio'] = prods_df['burst_ratio']
            self.grouped_df['burst_ids'] = prods_df['burst_ids']
            
            final_df['burst_ratio']=self.BST(self.grouped_df,self.Candidate_Groups)
            
            final_df['group_support']=self.Group_Support_Count(self.Candidate_Groups)
            
            final_df['group_size_ratio']=self.Group_Size_Ratio(self.raw_data_df,self.Candidate_Groups)
            
            final_df['group_content_similarity']=self.Group_Content_Similarity(self.raw_data_df,self.Candidate_Groups)
            
            gd=final_df['group_deviation'].tolist()
            rd=final_df['avg_rating_deviation'].tolist()
            bst=final_df['burst_ratio'].tolist()
            gsr=final_df['group_size_ratio'].tolist()
            gc=final_df['group_support'].tolist()
            gs=final_df['group_size'].tolist()
            rt=final_df['review_tightness'].tolist()
            gcs=final_df['group_content_similarity'].tolist()
            
            avg_score_groups=dict()
            for i in range(len(gd)):
                avg_score_groups[i]=(round((gd[i]+rd[i]+gcs[i]+gsr[i]+bst[i]+gc[i]+gs[i]+rt[i])/8,2))
            
            final_df['suspicious_score']=list(avg_score_groups.values())
            sort_by_score = final_df.sort_values('suspicious_score',ascending=False)
            data=self.getPredictions(self.raw_data_df,self.Candidate_Groups)
            return sort_by_score,data

        
    