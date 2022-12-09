import pandas as pd
from sklearn.model_selection import train_test_split
import re
from random import choices
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt



def get_column(column:str):
    counter = 0
    item_list = []
    for col in mydata:
        if column in col: #Looking columns containing the word "Brand"
            item_list.append(counter)
        counter = counter + 1
    return item_list

def make_brand_column(brand_at:list):
    # Empty Dataframe column
    brands = pd.DataFrame(index=range(len(mydata)),columns=range(1))
    
    # List of TV brandnames found on the internet to assign to the products, if this brand names is in the title
    brands_web = ["Dynex", "Insignia","Avue", "Venturer", "TCL", "Viore", "Elite", "CurtisYoung", "Azend", "Hiteker", "Contex", "ProScan", "GPX"]
    
    for j in range(len(mydata)):
        for i in brand_at: #Loop over the columns containing "Brand"
            if mydata.iloc[j,i] != 0: # If the element in this columns is not equal to zero
                brands.iloc[j,0] = mydata.iloc[j,i] # Add this element to the new created columnd at the given index
    # Set all nan values to zero, this makes it easier to compare
    brands = brands.fillna(0)
    
    #Create an empty list to add all brand names of the product found in a list
    brand_list = []
    
    # Adding all the given brand names into a brand list, containing no duplicates      
    for j in range(len(mydata)):
        brand_name = str(brands.iloc[j,0])
        if brand_name != '0':
            if brand_name not in brand_list:
                brand_list.append(brand_name)
   
    for i in range(len(brands_web)): 
        brand_list.append(brands_web[i])

    for j in range(len(mydata)):
        if brands.iloc[j,0] == 0:
            title = str(mydata.iloc[j,title_at])
            for i in range(len(brand_list)):
                if brand_list[i] in title:
                    brands.iloc[j,0] = brand_list[i]
                    
    for j in range(len(mydata)):
        name = brands.iloc[j,0]
        if "LG" in name:
            brands.iloc[j,0] = str("LG")
        
    return brands

class product_clean:
    def __init__(self, product_nr):
        self.product_nr = product_nr
                
    def get_product(self,product_nr):
        product = data_use.iloc[product_nr,:]
        return product
    
    def no_zero(self,product_nr):
        product = product_maker.get_product(product_nr)
        product = product[product!=0]
        return product

def get_inch(a):
    norm_inch = ['Inch', 'inches', '"', '-inch', ' inch', 'inch ', 'INCH', '-Inch', '”', "'" ]
    for x in norm_inch:
        if x in a:
            a = a.replace(x, "inch " )
    return a

def get_hz(a):
    norm_hz = ['Hertz','HZ','Hz', ' hz', '-hz', 'hz' ]
    for x in norm_hz:
        if x in a:
            a = a.replace(x, "hz " )
    return a  

def one_inch(word_list:list):
    counter = 0
    for word in word_list:
        if 'inch' in word:
            if counter > 0:
                word_list.remove(word)
            else:
                counter = 1
    return word_list

class lsh:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
    def get_signature_matrix_bands(self, full_list, bands_nr, n_hash): 
        # Calculate the number of rows in every band
        r =  int(n_hash/bands_nrs) 
        
        bands = {}
        for i in range(0,bands_nrs): # Make the amount of bands as prespecified
            bands[i] = []
        for signature in full_list: # We want to divide the signatures over the bands
            for i in range(0, bands_nrs):
                index = i*r    
                bands[i].append(signature[index:index+r])    
        return bands
    

    def get_band_buckets(self, band): 
        buckets = {}
        for doc_id in range(0,len(band)): 
            value = int(''.join(map(str, band[doc_id]))) # All the integers in the band I paste together to make unique name for
            if value not in buckets:                     # bukcet, when there is a bucket with the same name add them together
                buckets[value] = [doc_id]
            else:
                buckets[value].append(doc_id)      
        return buckets
    
    def get_candidates_list(self, buckets):
        candidates = set()
        # buckets is a dictionary containing key=bucket, value= list of doc_ids that hashed to bucket
        for bucket,candidate_list in buckets.items():
            if len(candidate_list) > 1:
                for i in range(0,len(candidate_list)-1): #Gelijk aan 0, meer niet
                    for j in range(i+1,len(candidate_list)):  # j=1, meer niet
                        pair = tuple(sorted( (candidate_list[i],candidate_list[j]) ))
                        candidates.add(pair)# werkt distinct
        return candidates #ie a set of couples, each couple is a candidate pair
  
    def get_similar_items(self, sig_matrix, bands_nr, sign_len):
        candidates_list = []
        #divide signature matrix into bands
        bands = lsh_instance.get_signature_matrix_bands(sig_matrix,bands_nr,sign_len)
        #for all the bands
        for band_id, elements in bands.items(): # Band ID is index, dus band 4 in de dictionariy heeft band ID 3, want je begint bij 0
            #produce the buckets for the given band (band_id) with a random hash function
            buckets = lsh_instance.get_band_buckets(elements)
            #Get all the candidate pairs
            candidates = lsh_instance.get_candidates_list(buckets)           
            candidates_list.append(candidates)
        return candidates_list


mydata = pd.read_excel(r"C:\Users\lotte\OneDrive\Bureaublad\Master\Periode 2\Computer Science for Business Analytics\Python\output.xlsx")
# Find the index for some specific columns with a definition
brand_at = get_column("Brand")
title_at = get_column("title")
feature_at = get_column("features")
shop_at = get_column("shop")

# I create an extra column, with the brand name of every product as value
brands = make_brand_column(brand_at)

# The location I want to add the column in the dataset, insert the new column and remove the other columns containing the word brand
col_brand = 0 
mydata = mydata.drop(mydata.columns[brand_at], axis=1)
mydata.insert(col_brand,'Brand', brands)

# Assign model_ID as Y, leave the rest as data to use
model_id = pd.DataFrame(mydata['modelID'])
mydata.drop('modelID', inplace=True, axis=1)

title_at = get_column("title")
feature_at = get_column("features")
shop_at = get_column("shop")

# Split the data in 63% training and 37% test set
bootstrap_list = []
plot_FOC_list =[]
plot_PC_list = []
plot_PQ_list = []
plot_F1_list = []
for k in range(5):
    data_train, data_test, model_ID_train, model_ID_test = train_test_split(mydata, model_id, test_size=0.37, shuffle = True)
    
    
    data_use = data_test
    y_use = model_ID_test
    
    print(y_use.iloc[1].item())
    
    index = data_train.index
    
    #--------------------------------------------------------------------------------------------------
    # Find all the duplicates in the train dataset
    y = y_use.values.tolist()
    n = len(y_use)
    
    duplicates = set()
    for i in range(n):
        a = y[i]
        for j in range(n):
            b = y[j]
            c = tuple(sorted((i,j)))
            if c not in duplicates:
                if a==b and i!=j:
                    duplicates.add(c)
    n = len(data_use)
    
    number_possible_pairs = 0
    for i in range(n):
        j = i+1
        number_possible_pairs = number_possible_pairs + (n-j)
    
    
    all_model_words = []            # List with all the modelwords together
    list_modelwords_lists = []      # Long list, containint 1624 lists, with in each lists the modelwords of 1 product
#------------------------------------------------------------------------------------------------------
    for i in range(n):
        product_model_words = set()
    
        product_maker = product_clean(i)
        product = product_maker.no_zero(i)
    
        brandname = product[col_brand].lower()
        product_model_words.add(brandname)
    
        if brandname not in all_model_words:
            all_model_words.append(brandname)
    
        title = product[title_at]
        title = ' '.join(map(str, title))               # Reorganize the title, to 1 string contaning several words
        title = get_inch(title)                         # Change everything to inch
        title = get_hz(title)                           # Change everything to hz
        title = re.sub(r"[^/-:.,a-zA-Z0-9 ]", "", title)# Remove strange characters from the title, but leave .,: numbers and letters in
    
        title_words = title.split()                     # Split the title in a list with words 
        title_words = one_inch(title_words)             # I want to get rid of the weird inch measures, so I remove them from the title
    
        for x in title_words:                           # Only keep the modelwords, containing letters and numbers
           if x.isalnum() and not x.isalpha() and not x.isdigit():
               if x not in all_model_words:             # And only append if it is not already in the list
                   all_model_words.append(x)
               if x not in product_model_words:         # And only append if it is not already in the list
                   product_model_words.add(x)
    
        # att_list = [] 
        # for j in range(feature_at[0], len(product)):    # Now we take the features of all the products and add them in the list
        #     att_list.append(product[j])
    
        # features = ' '.join(map(str, att_list))
        # features = get_inch(features)
        # features = get_hz(features)
        # features = re.sub(r"[^/-:.,a-zA-Z0-9 ]", "", features)
        # features_words = features.split()
    
        # for x in features_words:
        #     if x.isalnum() and not x.isalpha() and not x.isdigit():
        #         if x not in all_model_words:
        #             all_model_words.append(x)
        #         if x not in product_model_words:        # And only append if it is not already in the list
        #             product_model_words.add(x)
    
        list_modelwords_lists.append(product_model_words)

#-------------------------------------------------------------------------------------------------------
        # One hot encoding
    df_one_hot = pd.DataFrame([])
    i = 0
    for words_set in list_modelwords_lists:
        one_hot = [1 if x in words_set else 0 for x in all_model_words]
        df_one_hot[i] = one_hot
        i = i + 1

#---------------------------------------------------------------------------------------------------
# Minhashing
    number_of_modelwords =  len(all_model_words)
    n_hash = int(0.5*number_of_modelwords)
    
    # Minhashing 
    a = choices(range(1, 10000), k = n_hash)
    b = choices(range(1, 10000), k = n_hash)
    
    signature_matrix = pd.DataFrame(data = math.inf,index=range(n_hash),columns=range(n))          
    for rows in range(number_of_modelwords):
        print(rows)
        hash_list = []
        for hi in range(n_hash):
            x = a[hi]*rows + b[hi]
            h = x % n_hash
            hash_list.append(h)
    
        for i in range(n):
            if df_one_hot[i][rows] == 1:
                for hii in range(len(hash_list)):
                    if hash_list[hii] < signature_matrix[i][hii]:
                        signature_matrix[i][hii] = hash_list[hii]
    
    signature_list = []
    for i in range(n):
        small_list = []
        for j in range(n_hash):
            small_list.append(int(signature_matrix[i][j]))
        signature_list.append(small_list)

#----------------------------------------------------------------------------------------------------
    # Finding candidate pairs 
    
    
    band_score_list = []
    plot_FOC =[]
    plot_PC = []
    plot_PQ = []
    plot_F1 = []
    tel1 = 0
    
    r_list  = [20,22,24,26,28,30,32,34,36,38,40]
    for r_use in r_list:
        print(tel1)
        tel1 = tel1 + 1
        r =  r_use
        bands_nrs = int(n_hash/r)
        
                
        lsh_instance = lsh(0.8)
        pair_list = lsh_instance.get_similar_items(signature_list,bands_nrs,n_hash)


        distinct_pair_list = []
        count = 0
        for set_pair in pair_list:
            for pair in set_pair:
                if pair not in distinct_pair_list:
                    distinct_pair_list.append(pair) 
                count =  count + 1 
        
        cluster_pair = set()
        for pair in distinct_pair_list:
            id1 = int(pair[0])
            id2 = int(pair[1])
            
            one_hot1 = list(df_one_hot[id1])
            one_hot2 = list(df_one_hot[id2])
                        
            intersection = 0
            union = 0
            for i in range(len(a)):
                if one_hot1[i] == 1 and one_hot1[i] == one_hot2[i]:
                    intersection = intersection + 1
                    union = union + 1
                if one_hot1[i] == 1 and one_hot2[i] == 0:
                    union = union + 1
                if one_hot2[i] == 1 and one_hot1[i] == 0:
                    union = union + 1
            js = intersection /  union
            #js = len(one_hot1.intersection(one_hot2)) / len(one_hot1.union(one_hot2))
            jd = int(1-js)
            
            
            product1 = product_maker.no_zero(id1)
            product2 = product_maker.no_zero(id2)
            
            shopname1 = product1[shop_at]
            shopname2 = product2[shop_at]
            
            brandname1 = product1[col_brand].lower()
            brandname2 = product2[col_brand].lower()
            
            #Distance matrix for clustering 
            distance_matrix = np.zeros((2,2))
            
            if brandname1!=brandname2 or shopname1.equals(shopname2):
                distance_matrix[0,1] = 1
                distance_matrix[1,0] = 1
            else:
                distance_matrix[0,1] = jd
                distance_matrix[1,0] = jd
                
            ac = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single', compute_full_tree=True, distance_threshold=0.2)
            result = ac.fit(distance_matrix).labels_
            
          
            if result[0] ==  result[1]:
                cluster_pair.add(pair)
    
        fp = cluster_pair - duplicates
        tp = cluster_pair - fp
        fn = duplicates - cluster_pair
        
        recall = len(tp) / (len(tp) + len(fn))
        precision = len(tp)/ (len(tp) + len(fp))
        
        comparison_made = len(distinct_pair_list)
        FOC = comparison_made / number_possible_pairs      
        
        pair_quality = len(tp)/comparison_made
        pair_completeness = len(tp)/len(duplicates)
        
        f1_score = 2*(recall * precision) / (recall + precision)
        f1_star = 2*(pair_quality*pair_completeness) / (pair_quality + pair_completeness)     
        
        plot_FOC.append(FOC)
        plot_PC.append(pair_completeness)
        plot_PQ.append(pair_quality)
        plot_F1.append(f1_score)
        
        score_list = []
        score_list.append("Bands: ")
        score_list.append(bands_nrs)
        score_list.append("r: ")
        score_list.append(r)
        score_list.append("F1: ")
        score_list.append(f1_score)
        score_list.append("F1*: ")
        score_list.append(f1_star)
        score_list.append("FOC: ")
        score_list.append(FOC)
        score_list.append("Pq: ")
        score_list.append(pair_quality)
        score_list.append("Pc: ")
        score_list.append(pair_completeness)
        band_score_list.append(score_list)
    bootstrap_list.append(band_score_list)
    plot_FOC_list.append(plot_FOC)
    plot_PC_list.append(plot_PC)
    plot_PQ_list.append(plot_PQ)
    plot_F1_list.append(plot_F1)
    




# Take the average for plotting    
FOC1 = [sum(i) for i in zip(*plot_FOC_list)]
PC1 =  [sum(i) for i in zip(*plot_PC_list)]  
PQ1 =  [sum(i) for i in zip(*plot_PQ_list)]  
F11 =  [sum(i) for i in zip(*plot_F1_list)] 

FOC = []
for x in FOC1:
    FOC.append(float(x/5))

PC = []
for x in PC1:
    PC.append(float(x/5))

PQ = []
for x in PQ1:
    PQ.append(float(x/5))

F1 = []    
for x in F11:
    F1.append(float(x/5))
    


plt.plot(r,F1)
plt.axis([18,42, 0, 1])
plt.xlabel("Values of r")
plt.ylabel("F1-measure ")
plt.show()





