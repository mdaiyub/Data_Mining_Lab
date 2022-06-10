#!/usr/bin/env python
# coding: utf-8

# # Lab task   ID: 193-15-13526

# In[ ]:





# # Python Basics

# In[1]:


2 + 4 * 3


# In[2]:


2**4


# In[3]:


x = "Data " + "Mining " + "and " + "Machine" + " Learning"
print(x)


# In[4]:


x.upper()


# In[5]:


x.lower()


# In[6]:


len(x)


# In[7]:


a = 7 + 3

b = a * 4
c = b - 5
L = [a,b,c]
print(L)


# In[8]:


M = [15, 32, "d", 3.5]

N = L + M
print(N)


# In[9]:


len(N)


# In[10]:


N[3]


# In[11]:


N[:2]


# In[12]:


N[2:]


# In[13]:


N[3:6]


# In[14]:


# This is what a comment looks like

fruits = ['mango','oranges','litchi','bananas']
for fruit in fruits:
    print(fruit + ' for sale')
fruitPrices = {'mango': 2.00, 'oranges': 1.50, 'litchi': 2.75,'bananas': 1.5}
for fruit, price in fruitPrices.items():
    if price < 2.00:
        print('%s cost %f a pound' % (fruit, price))
    else:
        print(fruit + ' are too expensive!')


# In[15]:


# Example of python's list comprehension construction:

nums = [1,2,3,4,5,6]
plusOneNums = [x+1 for x in nums]
print("plusOneNums =", plusOneNums)
oddNums = [x for x in nums if x % 2 == 1]
print("oddNums = ", oddNums)
oddNumsPlusOne = [x+1 for x in nums if x % 2 ==1]
print("oddNumsPlusOn = ", oddNumsPlusOne)


# In[16]:


# Dictionaries

studentIds = {'knuth': 42.0, 'turing': 50.0, 'nash': 92.0 }
studentIds['turing']


# In[17]:


studentIds['ada'] = 97.0
studentIds


# In[18]:


del studentIds['knuth']
studentIds


# In[19]:


studentIds['knuth'] = [42.0,'forty-two']
studentIds


# In[20]:


studentIds.keys()


# In[21]:


studentIds.values()


# In[22]:


studentIds.items()


# In[23]:


len(studentIds)


# # Numpy

# In[24]:


import numpy as np


# In[25]:


# Creating a 1-d array
A = np.array([13,25,3,44,5])
A


# In[26]:


# Creating a 1-d array from an existing Python list

L = [11, 22, 23, 42, 55, 46]
A = np.array(L)
A


# In[27]:


A / 3


# In[28]:


np.array([1.0,2,3,4,5]) / 6


# In[29]:


M = np.array([2, 3, 4, 5, 6, 7])

P = L * M
P


# In[30]:


P.sum()


# In[31]:


P.mean()


# In[32]:


# Dot product of two vectors
print(L)
print(M)
np.dot(L, M) # works like sum of M[i]*L[i]


# In[33]:


# Creating a 2-d array
X = np.array([[1,2,3],[4,5,6],[7,8,9]])
X


# In[34]:


# Transpose a 2-d array
X.T


# In[35]:


# Array slicing
X[1,2]


# In[36]:


# Array slicing

X[2,2]


# In[37]:


X[:,1]


# In[38]:


X[0,:]


# In[39]:


X[1:,1:]


# In[40]:


X[1:,0:3]


# In[41]:


# Using arrays to index into other arrays (e.g., Boolean masks)

A = np.array([1, 2, 3, 4, 5, 6])
A > 3


# In[42]:


A[A > 3]


# In[43]:


sum(A[A > 3])


# In[44]:


Ind = (A > 1) & (A < 6)
print(Ind)
A[Ind]


# # Numpy Part-2

# In[45]:


import numpy as np


# In[46]:


data = np.genfromtxt("D:\Academic\8th semester\Data Mining Lab\CSV\Video_Store.csv", delimiter=",", dtype=None)
data


# In[47]:


labels = data[0]
print(labels)


# In[48]:


data = data[1:]
data[0:5]


# **Now we can convert columns to the appropriate type as necessary:**

# In[49]:


age = np.array(data[:,3], dtype=int)
print(age)


# In[50]:


sal = np.array(data[:,2], dtype=float)
print(sal)


# In[51]:


min_sal = sal.min()
max_sal = sal.max()
print(min_sal, max_sal)


# In[52]:


visit_avg = np.array(data[:,5], dtype=float)
rentals = np.array(data[:,4], dtype=float)
print(visit_avg)
print(rentals)


# In[53]:


norm_sal = [(x-min_sal)/(max_sal-min_sal) for x in sal]

print(norm_sal)


# In[54]:


np.set_printoptions(precision=4, linewidth=80, suppress=True)

sal_range = max_sal - min_sal
norm_sal = (sal - min_sal) / sal_range
print(norm_sal)


# **Z-Score Standardization on Age**

# In[55]:


age_mean = age.mean()

age_std = age.std()
print(age_mean, age_std)


# In[56]:


age_znorm = (age - age_mean) / age_std
print(age_znorm)


# **Suppose that we would like to find all "good cutomers" defined as those with Rentals value
# of >= 30:**

# In[57]:


is_good = np.array(rentals >= 30)
good_cust = np.array(data[is_good])
print(good_cust)


# **Now, suppose we want to change the Gender atrribute into binary (converting it from one
# cateogrical attribute with two values into two attributes (say 'Gender_F' and 'Gender_M')
# with 0 or 1 as values depending on the original value. These new variables are sometimes
# called "dummy" variable. The purpose of this transfromation is to allow for the application
# of technqiues that require all attributes to be numerical (e.g., regression analysis or
# correlation analysis). Below, we show how this could be done manually for illustration
# purposes. In practice (as we shall see later in this notebook), there are Python libraries and
# packages that perform this type of transformation automatically.**

# In[58]:


gender = np.array(data[:,1])
gender


# In[59]:


gen_f = np.zeros(len(gender))
gen_f


# In[60]:


gen_f[gender==b'F'] = 1
gen_f


# In[61]:


gen_m = np.zeros(len(gender))

gen_m[gender==b'M'] = 1
gen_m


# **Let's now create a new 2d array with the old Gender attributes replaced with the new ones.
# In the example below, we have removed the two other categorical attributes (Incidentals and
# Genre) for now, just to illustrate what the data would look like in "Standard Spreadsheet
# Fromat":**

# In[62]:


vs_new = np.array([gen_f,gen_m,sal,age,rentals,visit_avg])

vs_new = vs_new.T
np.set_printoptions(linewidth=80)
#Here are the first 5 elements of the new array
print(vs_new[0:15])


# **we can apply standard statistical or numeric functions to the whole array and not
# just to individual columns:**

# In[63]:


f_mean,m_mean,sal_mean,age_mean,rentals_mean,visavg_mean= vs_new.mean(axis=0)

print(" Gen=F Gen=M Income Age Rntls VisAvg")
print("Mean: ", f_mean,m_mean,sal_mean,age_mean,rentals_mean,visavg_mean)


# In[64]:


#Now that the data is in all numeric form, we can apply techiques such as correlation analysis

np.corrcoef(vs_new.T)


# In[65]:


# The new table can be written into a file using "savetxt" function:
out_file = open("D:/Academic/8th semester/Data Mining Lab/new/new_file", "w")

np.savetxt(out_file, vs_new, fmt='%d,%d,%1.2f,%1.2f,%1.2f,%1.2f', delimiter='\t')


# **An alternative method for loading heterogenous (mixed type) data into an array is to specify
# the dtype and set "Names" to "True". This creates a structured array with each row
# representing a tuple. Each column can be accessed by the keys extracted from the first line
# of the data file.**

# In[66]:


vs = np.genfromtxt("D:\Academic\8th semester\Data Mining Lab\CSV\Video_Store.csv", delimiter=",", names=True, dtype=('S1','S1',int,int,int,float,'S10'))

print(vs)


# In[67]:


np.dtype(vs[0])


# In[68]:


print(vs['Gender'])


# In[69]:


print(vs['Income'])


# In[70]:


is_good = np.array(vs['Rentals'] >= 30)
good_cust = np.array(vs[is_good])
good_cust


# In[71]:


print("Min Rentals: ", good_cust['Rentals'].min())
print("Max Rentals: ", good_cust['Rentals'].max())
print("Rentals Mean: ", good_cust['Rentals'].mean())
print("Rentals Median: ", np.median(good_cust['Rentals']))
print("Rentals Std. Dev.: ", good_cust['Rentals'].std())


# **For most types of analysis, we would want to exclude the index column from the data (in
# this case the ID attribute). This could be done by removing the first column of the matrix.
# However, it could have been done when reading the data in using the "usecols" parameter
# in "genfromtxt".**

# In[72]:


vs_nid = np.genfromtxt("D:\Academic\8th semester\Data Mining Lab\CSV\Video_Store.csv", delimiter=",", usecols=(1,2,3,4,5,6), dtype=('S1','S1','S10','S10','S10','S10','S10'))
vs_nid[0:5]


# **Once the data is in structured array format as above, we can combine the tuples with
# feature names to create an array of dicts. The DictVectorizer package from the Scikit-learn
# library can then be used to create dummy variables for each of the categorical attriibutes
# and convert the data into the standard spreadsheet format. This is the preferred approach
# for creating dummy variables than the manual approach discussed earlier in cells
# 25-30.**

# In[73]:


names = vs_nid.dtype.names

vs_dict = [dict(zip(names, record)) for record in vs_nid]


# In[74]:


print(vs_dict[0])


# In[ ]:





# In[ ]:





# In[75]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


plt.hist(vs['Income'], bins=9, alpha=0.5)

plt.xlabel('Income')
plt.ylabel('Count')
plt.title('Histogram of Income')
plt.axis([0, 100000, 0, 10])
plt.grid(True)
plt.show()


# **cross-tabulate the Genre and the Gender attributes to find out if men and women have
# different movie preferences. [Note: correlation analysis perfromed earlier could also shed
# some light on this question.]**

# In[77]:


# First we need the counts for males and females across different genres
m_counts = [14, 6, 8] # counts of Action, Comedy, Drama for male custs.
f_counts = [8, 6, 12] # counts of Action, Comedy, Drama for female custs.
N = len(f_counts)


# In[78]:


ind = np.arange(N) # the x locations for the groups

ind = ind + 0.15
width = 0.35 # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, f_counts, width, color='g')
rects2 = ax.bar(ind+width, m_counts, width, color='r')
rects1 = plt.bar(ind, f_counts, width, color='g')
rects2 = plt.bar(ind+width, m_counts, width, color='r')
ax.set_ylabel('Counts')
ax.set_ybound(upper=16)
ax.set_title('Counts by Genre and Gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Action', 'Comedy', 'Drama') )
ax.legend( (rects1[0], rects2[0]), ('Female', 'Male') )
plt.show()
# plt.savefig("figure.pdf")


# This figure shows that male customers tend to prefer action movies, while female
# customers tend to like dramas.

# In[ ]:





# **scatter plot discover possible correlations between Age and Income.**

# In[79]:


fig = plt.figure(figsize=(5, 4))

# Create an Axes object.
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
# Plot the data.
ax.scatter(vs['Age'], vs['Income'], color="green", marker="*")
# Add a title.
ax.set_title("Age VS. Income")
# Add some axis labels.
ax.set_xlabel("Age")
ax.set_ylabel("Income")
# Produce an image.
# fig.savefig("scatterplot.png")
plt.show()


# Indeed, there does appear to be a positive correlation between Age and Income. This can
# also be observed by viewing the Pearson r correlation coefficient in our correlation matrix
# shown earlier.

# In[ ]:





# In[ ]:





# # Working With Population.txt data file

# In[80]:


datafile = open("D:/Academic/8th semester/Data Mining Lab/txt/populations.txt", "r")
#first line of the file contains labels for column names
labels_line = datafile.readline().strip()
poptable = []
total = 0
for line in datafile.readlines():
    line = line.strip()
    row = line.split("\t")
    record = [int(row[0]),float(row[1]),float(row[2]), float(row[3])]
    print(record)
    poptable.append(record)
    total += 1
print(total)


# In[81]:


print(poptable)


# In[82]:


#printing the top5 elements
print(poptable[0:5])


# **Following is an example of creating Python dictionaries for each of the species in the data.
# The year (in the first) column is used as the key for the population value of each of the
# species for that year.**

# In[84]:


hares = {}

lynxes = {}
carrots = {}
for row in poptable:
    hares[row[0]] = row[1]
    lynxes[row[0]] = row[2]
    carrots[row[0]] = row[3]
print (sorted(hares.items(), key=hares.get))


# In[ ]:





# In[85]:


print(hares[1903])


# In[86]:


# finding the years during which the population of hares was greater than 50K
# Here we'll use standard Python list comprehensions

hares_above_50k = [yr for yr in hares.keys() if hares[yr]>50000.0]
print(hares_above_50k)


# In[87]:


# Finding the year(s) with maximal value of Hares

maxhares = [yr for yr in hares.keys() if hares[yr] == max(hares.values())]
for i in range(0,len(maxhares)):
    print(maxhares[i], hares[maxhares[i]])


# **So far, we have not used Numpy, only standard Python. But, many operations on data
# involving tables or matrices are much simpler and more efficient using Numpy. Let's now
# try using NumPy arrays:**

# In[88]:


import numpy as np


# In[89]:


pop = np.array(poptable)
print(pop)


# **However, we could have, from the start, loaded the data into a Numpy array. One way to do
# this is via "loadtxt" which reads numerical values from tab-delimited or CSV files (converted
# into floats).**

# In[90]:


poptable = np.loadtxt('D:/Academic/8th semester/Data Mining Lab/txt/populations.txt', skiprows=1)

print(poptable)


# In[91]:


poptable.shape


# In[ ]:





# **In fact, we can assign each column directly into separate 1-d arrays, by splitting the
# transpose of poptable:**

# In[92]:


year, hares, lynxes, carrots = poptable.T
print(year)
print(hares)
print("Mean Hare Population: ", hares.mean())

# Numpy allows us to easily perform operations on rows, columns, or to the whole array:


# In[93]:


# finding all years when the population of one of the species is above 50k

above_50k = np.any(poptable>50000, axis=1) # axis=1 means the operation will be performed
print(above_50k)
print(year[above_50k])


# In[94]:


# now print some summary statistics on each of the species:

pop_no_year = poptable[:,1:] # Removing the first column ("Year")

print (" Hares Lynxes Carrots")
print ("Mean:", pop_no_year.mean(axis=0))
print ("Std: ", pop_no_year.std(axis=0))


# In[95]:


# Finding indecies of years when one of the populations was at max

j_max_years = np.argmax(pop_no_year, axis=0) # ranging over rows for each column
print ("Indecies for the maximums:", j_max_years)
print(" Hares Lynxes Carrots")
print("Max. year:", year[j_max_years])


# In[96]:


# Ranging over cols for each row, find the specie with the highest pop for each year

max_species = np.argmax(pop_no_year, axis=1)
species = np.array(['Hare', 'Lynx', 'Carrot'])
print(max_species)
print("Max specie from 1900 to 1920:")
print(species[max_species])


# **correlations among the three population variables**

# In[97]:


corr_matrix = np.corrcoef(pop_no_year.T)
print(corr_matrix)


# In[98]:


pop_with_keys = poptable.view(dtype=[('year', 'float'), ('hares', 'float'),('lynx', 'float'),('carrot', 'float')])
pop_with_keys


# In[99]:


print(pop_with_keys['hares'])


# In[100]:


# Now we can do sorting using the desired label. For example, we can sort the table using the 'hares' field:

sorted_by_hares = np.sort(pop_with_keys, order='hares', axis=0)
print(sorted_by_hares)


# In[ ]:





# **basic visualization with matplotlib:**

# In[101]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


plt.plot(year, hares)


# In[103]:


plt.plot(year, hares, label='Hares')
plt.plot(year, lynxes, label='Lynxes')
plt.plot(year, carrots, label='Carrots')
plt.legend( ('Hares','Lynxes','Carrots') )
plt.ylabel('Population')
plt.xlabel('Year')
plt.show()


# In[104]:


plt.hist(carrots, bins=8, alpha=0.5)

plt.xlabel('Carrots')
plt.ylabel('Count')
plt.title('Histogram of Carrot Populaions')
plt.axis([35000, 50000, 0, 6])
plt.grid(True)


# In[105]:


plt.scatter(hares, carrots, color="blue", marker="*")

plt.xlabel('Hares')
plt.ylabel('Carrots')
plt.title('Hares v. Carrots')
plt.grid(True)


# In[106]:


fig = plt.figure(figsize=(12, 4))

# Create an Axes object.
ax1 = fig.add_subplot(1,2,1) # one row, two column, first plot
# Plot the data.
ax1.scatter(hares, carrots, color="red", marker="*")
ax1.set_title("Hares vs. Carrots")
# Add some axis labels.
ax1.set_xlabel("Hare Population")
ax1.set_ylabel("Carrot Population")
ax2 = fig.add_subplot(1,2,2) # one row, two column, 2nd plot
# Plot the data.
ax2.scatter(hares, lynxes, color="blue", marker="^")
ax2.set_title("Hares vs. Lynxes")
# Add some axis labels.
ax2.set_xlabel("Hare Population")
ax2.set_ylabel("Lynx Population")
# Produce an image.
# fig.savefig("scatterplot.png")


# In[ ]:





# In[ ]:




