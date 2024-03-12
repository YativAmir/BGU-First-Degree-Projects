import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pycountry
import seaborn as sns
from sklearn.metrics import jaccard_score
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from collections import Counter
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from scipy.stats import chi2_contingency
import statsmodels.graphics.mosaicplot as mosaic
from statsmodels.graphics.mosaicplot import mosaic
import pycountry as pc
import pycountry_convert as pcc
import statistics
import re
import math
import nltk
from collections import Counter
#nltk.download('averaged_perceptron_tagger')
#import geopandas as gpd
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import stopwords
#nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize, sent_tokenize

#--------------------------------------- read csv file:
table = pd.read_csv('Xy_train.csv')
table = table.replace(np.nan, " ", regex=True )
headers = table.columns.tolist()  # Get the column names as a list
inf = float('inf')
fraudulent_table = table[table['fraudulent'] == 1]
no_fraudulent_table = table[table['fraudulent'] == 0]
#
# create a sample dataset
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# create a PCA object with 2 components
pca = PCA(n_components=2)

# fit the data to the PCA object
pca.fit(X)

# transform the data using the PCA object
X_pca = pca.transform(X)

# print the transformed data
print(X_pca)


def without_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 12)

#------------------------------------------------title---------------------------------------------
#Group the DataFrame by the fraudulent column and get the binary values
text_data = table.groupby('fraudulent')
unique_binary_vals = text_data.groups.keys()
# Create two empty lists to store the grouped string values
true_fraudulent = []
false_fraudulent = []
# Iterate over the binary values and append the corresponding string values to the appropriate list
for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'title'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)
# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_false]
# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])
# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(20)
most_common_company_profile_false = fdist_false.most_common(20)

# creat a dictionary which k=word and v=count
most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}

count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}
# Iterate over the title column count the word that appears in the 100 common words
for index, row in table.iterrows():
    sentence = str(row['title'])
    words =  word_tokenize(sentence)  # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Title where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Title fraudulent=1")
plt.xlabel("Count")
plt.ylabel("20 common words")
plt.show()
# -------------------------------------represetation
def wordRating(GW, BW, word):
    if word in GW:
        return 0
    if word in BW:
        return 1
    return 0.5

titleGW=list(set(labels_true) - set(labels_false))
titleBW=list(set(labels_false) - set(labels_true))

table.insert(3, column="title_not_fraudulent", value=0)  # create new column that shows how many words in the description
table.insert(3, column="title_fraudulent", value=0)  # create new column that shows how many words in the description
table.insert(3, column="title_ratio", value=0)  # create new column that shows how many words in the description


for index, row in table.iterrows():
    count_not_frau = 0
    count_frau = 0
    sentence = str(row['title'])
    tokenized_data = tokens = nltk.word_tokenize(sentence)
    for word in tokenized_data:
        if word in titleGW:
            count_not_frau = count_not_frau+1
            table.loc[index, 'title_not_fraudulent'] = count_not_frau  # update the column
        if word in titleBW:
            count_frau = count_frau+1
            table.loc[index, 'title_fraudulent'] = count_frau # update the column
    # if int(row['title_not_fraudulent']) == 0 and int(row['title_fraudulent']) == 0:
    #     table.loc[index, 'title_ratio'] = 0.5
    # else:
    #     table.loc[index, 'title_ratio'] = row['title_not_fraudulent'] / row['title_not_fraudulent'] + row[
    #         'title_fraudulent']  # update the column


# ------------------------------------------------location---------------------------------------------
def get_continent(country_name, null=None):
    try:
        country = pc.countries.get(name=country_name)
        continent_code = pcc.country_alpha2_to_continent_code(country.alpha_2)
        continent_name = pcc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return null

def get_country():
    count = 0
    for index, row in table.iterrows():
        if (row['location']==' '):
            table.loc[index, 'continent'] = " "  # update the column
            table.loc[index, 'country'] = " "  # update the column
        else:
            country = row['location'][:2]
            country = pc.countries.get(alpha_2=country)
            table.loc[index, 'continent'] = get_continent(country.name)  # update the column
            table.loc[index, 'country'] = country.name  # update the column

# read csv file:
count = 0
headers = table.columns.tolist()  # Get the column names as a list
table.insert(3, column="continent", value=0)  # create new column that shows witch continent is the job in
table.insert(4, column="country", value=0)  # create new column that shows witch country is the job in
get_country()

# Create an empty dictionary to store the country counts
country_counts = {}

# Iterate over each country in the list
for index, row in table.iterrows():
    # If the country is already in the dictionary, increment its count by 1
    if row['country'] in country_counts:
        country_counts[row['country']] += 1
    # If the country is not in the dictionary, add it with a count of 1
    else:
        country_counts[row['country']] = 1

# Print the country counts
for country, count in country_counts.items():
    # print(country, count)

    continent_counts = {}

# Iterate over each continent in the list
for index, row in table.iterrows():
    # If the continent is already in the dictionary, increment its count by 1
    if row['continent'] in continent_counts:
        continent_counts[row['continent']] += 1
    # If the continent is not in the dictionary, add it with a count of 1
    else:
        continent_counts[row['continent']] = 1

    # Print the continent counts
#for continent, count in continent_counts.items():
# print(continent, count)

sns.countplot(x='continent', hue='continent', dodge=False, data=table)
plt.ylabel('Count', size=15)
plt.xlabel('continent', size=15)
plt.title("jobs by continent", size=15, weight='bold')
#plt.legend(labels=['0', '1'], loc=(0.75, 0.8),  title='fraudulent')
plt.show()

fraudulent_table = table[table['fraudulent'] == 1]
sns.countplot(x='continent', hue='continent', dodge=False, data=fraudulent_table)
plt.ylabel('Count', size=15)
plt.xlabel('continent', size=15)
plt.title("Jobs By Continent for Fraudulent Job", size=15, weight='bold')
#plt.legend(labels=['0', '1'], loc=(0.75, 0.8),  title='fraudulent')
plt.show()
# plt.figure(figsize=(10,6))
# sns.countplot(x='country', data=table, hue="fraudulent", order=table['country'].value_counts().iloc[:10].index)
# plt.xticks(rotation=90)
# plt.show()
# -----------------------------------------complete country


# for index, row in table.iterrows():
#     if row['location'] == ' ':
#         sentence1 = row['title']#[nltk.word_tokenize(text) for text in row['title']]
#         for country in pycountry.countries:
#             if country.name in sentence1:
#                 table.loc[index, 'location'] = pycountry.countries.get(name=country.name).alpha_2
#                 table.loc[index, 'country'] = country.name
#                 table.loc[index, 'continent'] = get_continent(country.name)
# p=3


#------------------------------------------------department---------------------------------------------
# Group the DataFrame by the fraudulent column and get the binary values
text_data = table.groupby('fraudulent')
unique_binary_vals = text_data.groups.keys()
# Create two empty lists to store the grouped string values
true_fraudulent = []
false_fraudulent = []
# Iterate over the binary values and append the corresponding string values to the appropriate list
for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'department'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)
# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]
# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_false]
# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])
# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(20)
most_common_company_profile_false = fdist_false.most_common(20)
# creat a dictionary which k=word and v=count
most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}

count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}
# Iterate over the title column count the word that appears in the 100 common words
for index, row in table.iterrows():
    sentence = str(row['department'])
    words =  word_tokenize(sentence)  # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Department where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Department fraudulent=1")
plt.xlabel("Count")
plt.ylabel("20 common words")
plt.show()


# --------------------------------------------mean of salary Range
table.insert(7, column="mean_salary_range", value=0)  # create new column that shows how many words in the description
for index, row in table.iterrows():
    mean = 0
    number_range = row['salary_range']
    number_list = number_range.split("-")  # calculate how many words
    if number_range != ' ':
        for element in number_list:
            if "?" in element:
                for element in number_list:
                    if element.isdigit():
                        mean = int(element)
                        table.loc[index, 'mean_salary_range'] = mean  # update the column
            else:
                continue
        if len(number_list) == 1:
            for element in number_list:
                mean = int(element)
                table.loc[index, 'mean_salary_range'] = mean  # update the column
        if mean == 0:
            lower_bound = int(number_list[0])
            upper_bound = int(number_list[1])
            mean = statistics.mean([lower_bound, upper_bound])
            table.loc[index, 'mean_salary_range'] = mean  # update the column
#max_age = max(table['mean_salary_range'])

# define the bin edges for each category
mean_salary_range_bins = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, inf]

# use pd.cut() to categorize the numbers based on the bin edges
categories = pd.cut(table['mean_salary_range'], bins=mean_salary_range_bins)

# create a dataframe with the categories
mean_salary_range_categories = pd.DataFrame({'Numbers': table['mean_salary_range'], 'Categories': categories})

# plot a countplot using seaborn to visualize the distribution of the categories
men_plot=sns.countplot(x='Categories', hue='Categories', dodge=False, data=mean_salary_range_categories)
plt.ylabel('Count', size = 13)
plt.xlabel('mean salary range Bins', size=15)
plt.title("mean salary range Bins", size = 15, weight='bold')
plt.legend(labels=['0-20000', '20000-40000', '40000-60000', '60000-80000', '80000-100000', '100000-120000',
                   '120000-140000', '140000-160000', '160000-180000', '180000-200000', '200000-inf'],
           loc=(0.6, 0.3), title='categorized num of words')
#without_hue(numPrevOwnersPlot1111, table['company_profile_num_of_words'])
plt.show()

# ---------------------------mean_salary_normalized




# ------------------------------------------------------company_profile---------------------------------------------------------
text_data1 = table.groupby('fraudulent')
unique_binary_vals = text_data1.groups.keys()
true_fraudulent = []
false_fraudulent = []

for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data1.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'company_profile'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)

# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_false]

# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])


# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(10)
most_common_company_profile_false = fdist_false.most_common(10)

most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}


# Apply the function to the column and create a new column for the top word counts
count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}

for index, row in table.iterrows():
    sentence = str(row['company_profile'])
    words = word_tokenize(sentence)  # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("10 common words in Company Profile where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("10 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("10 common words in Company Profile where fraudulent=1")
plt.xlabel("Count")
plt.ylabel("10 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

table.insert(6, column="company_profile_num_of_words", value=0)  # create new column that shows how many words in the description
for index, row in table.iterrows():
    number = len(sentence.split())  # calculate how many words
    table.loc[index, 'company_profile_num_of_words'] = number  # update the column
company_profile_numbers = table['company_profile_num_of_words']
# define the bin edges for each category
company_profile_numbers_bins = [0, 100, 200, 300, 400, 500, inf]

# use pd.cut() to categorize the numbers based on the bin edges
categories = pd.cut(company_profile_numbers, bins=company_profile_numbers_bins)

# create a dataframe with the categories
num_word_des_categories = pd.DataFrame({'Numbers': company_profile_numbers, 'Categories': categories})

# plot a countplot using seaborn to visualize the distribution of the categories
sns.countplot(x='Categories', hue='Categories', dodge=False, data=num_word_des_categories)
plt.ylabel('Count', size = 13)
plt.xlabel('Number of Words Bins', size=15)
plt.title("Number of Words: Company Profile", size = 15, weight='bold')
plt.legend(labels=['0-100', '100-200', '200-300', '300-400', '400-500', '500-inf'],
           loc=(0.6, 0.3), title='categorized num of words')
#without_hue(numPrevOwnersPlot1111, table['company_profile_num_of_words'])
plt.show()

# ------------------------------------------num of word feature representation
table.insert(6, column="com_pro_normalized_num_of_words", value=0)
# find the maximum value in the column
max_value = table['company_profile_num_of_words'].max()
for index, row in table.iterrows():
    value = row['company_profile_num_of_words']# calculate how many words
    normalized = value/max_value
    table.loc[index, 'com_pro_normalized_num_of_words'] = normalized  # update the column




# ------------------------------------------------------description---------------------------------------------------------
text_data1 = table.groupby('fraudulent')
unique_binary_vals = text_data1.groups.keys()
true_fraudulent = []
false_fraudulent = []

for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data1.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'description'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)


# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_false]

# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])


# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(20)
most_common_company_profile_false = fdist_false.most_common(20)

most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}


# Apply the function to the column and create a new column for the top word counts
count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}

for index, row in table.iterrows():
    sentence = str(row['description'])
    words =  word_tokenize(sentence) # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Description where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Description where fraudulent=1")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

table.insert(8, column="description_num_of_words", value=0)  # create new column that shows how many words in the description
for index, row in table.iterrows():
    sentence = str(row['description'])
    number = len(sentence.split())  # calculate how many words
    table.loc[index, 'description_num_of_words'] = number  # update the column
company_profile_numbers = table['description_num_of_words']
# define the bin edges for each category
company_profile_numbers_bins = [0, 100, 200, 300, 400, 500, 600, 700, inf]

# use pd.cut() to categorize the numbers based on the bin edges
categories = pd.cut(company_profile_numbers, bins=company_profile_numbers_bins)

# create a dataframe with the categories
num_word_des_categories = pd.DataFrame({'Numbers': company_profile_numbers, 'Categories': categories})

# plot a countplot using seaborn to visualize the distribution of the categories
numPrevOwnersPlot1111=sns.countplot(x='Categories', hue='Categories', dodge=False, data=num_word_des_categories)
plt.ylabel('Count', size = 13)
plt.xlabel('Number of Words Bins', size=15)
plt.title("Number of Words: description", size = 15, weight='bold')
plt.legend(labels=['0-100', '100-200', '200-300', '300-400', '400-500', '500-600',
                   '600-700', '700-inf'],
           loc=(0.6, 0.3), title='categorized num of words')
#without_hue(numPrevOwnersPlot1111, table['company_profile_num_of_words'])
plt.show()

#----------------representation
table.insert(6, column="des_normalized_num_of_words", value=0)
# find the maximum value in the column
max_value = table['description_num_of_words'].max()
for index, row in table.iterrows():
    value = row['description_num_of_words']# calculate how many words
    normalized = value/max_value
    table.loc[index, 'des_normalized_num_of_words'] = normalized  # update the column



# ------------------------------------------------------requirements---------------------------------------------------------
text_data1 = table.groupby('fraudulent')
unique_binary_vals = text_data1.groups.keys()
true_fraudulent = []
false_fraudulent = []

for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data1.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'requirements'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)

# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation and word.lower() not in filtered_data_true] for text in tokenized_data_false]

# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])


# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(20)
most_common_company_profile_false = fdist_false.most_common(20)

most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}


# Apply the function to the column and create a new column for the top word counts
count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}

for index, row in table.iterrows():
    sentence = str(row['requirements'])
    words = word_tokenize(sentence)  # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Requirements where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Requirements where fraudulent=1")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

table.insert(10, column="requirements_num_of_words", value=0)  # create new column that shows how many words in the description
for index, row in table.iterrows():
    sentence = str(row['requirements'])
    number = len(sentence.split())  # calculate how many words
    table.loc[index, 'requirements_num_of_words'] = number  # update the column
company_profile_numbers = table['requirements_num_of_words']
# define the bin edges for each category
company_profile_numbers_bins = [0, 100, 200, 300, 400, 500, inf]

# use pd.cut() to categorize the numbers based on the bin edges
categories = pd.cut(company_profile_numbers, bins=company_profile_numbers_bins)

# create a dataframe with the categories
num_word_des_categories = pd.DataFrame({'Numbers': company_profile_numbers, 'Categories': categories})

# plot a countplot using seaborn to visualize the distribution of the categories
numPrevOwnersPlot1111=sns.countplot(x='Categories', hue='Categories', dodge=False, data=num_word_des_categories)
plt.ylabel('Count', size = 13)
plt.xlabel('Number of Words Bins', size=15)
plt.title("Number of Words: requirements", size = 15, weight='bold')
plt.legend(labels=['0-100', '100-200', '200-300', '300-400', '400-500', '500-inf'],
           loc=(0.6, 0.3), title='categorized num of words')
#without_hue(numPrevOwnersPlot1111, table['company_profile_num_of_words'])
plt.show()

#----------------representation
table.insert(6, column="req_normalized_num_of_words", value=0)
# find the maximum value in the column
max_value = table['req_normalized_num_of_words'].max()
for index, row in table.iterrows():
    value = row['requirements_num_of_words']# calculate how many words
    normalized = value/max_value
    table.loc[index, 'com_pro_normalized_num_of_words'] = normalized  # update the column



# ------------------------------------------------------benefits---------------------------------------------------------
text_data1 = table.groupby('fraudulent')
unique_binary_vals = text_data1.groups.keys()
true_fraudulent = []
false_fraudulent = []

for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data1.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'benefits'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)

# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_false]

# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])


# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(20)
most_common_company_profile_false = fdist_false.most_common(20)

most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}


# Apply the function to the column and create a new column for the top word counts
count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}

for index, row in table.iterrows():
    sentence = str(row['benefits'])
    words = word_tokenize(sentence)  # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Benefits where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in Benefits where fraudulent=1")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

table.insert(12, column="benefits_num_of_words", value=0)  # create new column that shows how many words in the description
for index, row in table.iterrows():
    sentence = str(row['benefits'])
    number = len(sentence.split())  # calculate how many words
    table.loc[index, 'benefits_num_of_words'] = number  # update the column
company_profile_numbers = table['benefits_num_of_words']
# define the bin edges for each category
company_profile_numbers_bins = [0, 100, 200, 300, 400, 500, inf]

# use pd.cut() to categorize the numbers based on the bin edges
categories = pd.cut(company_profile_numbers, bins=company_profile_numbers_bins)

# create a dataframe with the categories
num_word_des_categories = pd.DataFrame({'Numbers': company_profile_numbers, 'Categories': categories})

# plot a countplot using seaborn to visualize the distribution of the categories
numPrevOwnersPlot1111=sns.countplot(x='Categories', hue='Categories', dodge=False, data=num_word_des_categories)
plt.ylabel('Count', size = 13)
plt.xlabel('Number of Words Bins', size=15)
plt.title("Number of Words: benefits", size = 15, weight='bold')
plt.legend(labels=['0-100', '100-200', '200-300', '300-400', '400-500', '500-inf'],
           loc=(0.6, 0.3), title='categorized num of words')
#without_hue(numPrevOwnersPlot1111, table['company_profile_num_of_words'])
plt.show()

#----------------representation
table.insert(6, column="ben_normalized_num_of_words", value=0)
# find the maximum value in the column
max_value = table['benefits_num_of_words'].max()
for index, row in table.iterrows():
    value = row['benefits_num_of_words']# calculate how many words
    normalized = value/max_value
    table.loc[index, 'ben_normalized_num_of_words'] = normalized  # update the column



# ------------------------------------------------------telecommuting---------------------------------------------------------
telecommuting_Plot1 = sns.countplot(x='telecommuting', hue='telecommuting', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Telecommuting', size = 13)
plt.title("Existence Of Telecommuting", size = 15, weight='bold')
plt.legend(labels=['No Tel', 'Has Tel'], loc=(0.75, 0.8),  title='Telecommuting')
without_hue(telecommuting_Plot1, table['telecommuting'])
plt.show()

telecommuting_Plot2 = sns.countplot(x='telecommuting', hue='telecommuting', dodge=False,  data=fraudulent_table)
plt.ylabel('Count', size = 13)
plt.xlabel('Has Company Logo', size = 13)
plt.title("Existence Of Telecommuting for Fraudulent", size = 15, weight='bold')
plt.legend(labels=['No Tel', 'Has Tel'], loc=(0.6, 0.8),  title='Telecommuting')
without_hue(telecommuting_Plot2, fraudulent_table['telecommuting'])
plt.show()

# ------------------------------------------------------has_company_logo---------------------------------------------------------
has_company_logo_Plot1 = sns.countplot(x='has_company_logo', hue='has_company_logo', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Has Company Logo', size = 13)
plt.title("Existence Of Company Logo", size = 15, weight='bold')
plt.legend(labels=['No Logo', 'Has Logo'], loc=(0.05, 0.8),  title='Company Logo')
without_hue(has_company_logo_Plot1, table['has_company_logo'])
plt.show()

has_company_logo_Plot2 = sns.countplot(x='has_company_logo', hue='has_company_logo', dodge=False,  data=fraudulent_table)
plt.ylabel('Count', size = 13)
plt.xlabel('Has Company Logo', size = 13)
plt.title("Existence Of Company Logo for Fraudulent", size = 15, weight='bold')
plt.legend(labels=['No Logo', 'Has Logo'], loc=(0.6, 0.8),  title='Company Logo')
without_hue(has_company_logo_Plot2, fraudulent_table['has_company_logo'])
plt.show()

# ------------------------------------------------------has_questions---------------------------------------------------------
questions_Plot1 = sns.countplot(x='has_questions', hue='has_questions', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Has Questions', size = 13)
plt.title("Existence Of a questions", size = 15, weight='bold')
plt.legend(labels=['No questions', 'Has questions'], loc=(0.05, 0.8),  title='questions')
without_hue(questions_Plot1, table['has_questions'])
plt.show()

questions_Plot2 = sns.countplot(x='has_questions', hue='has_questions', dodge=False,  data=fraudulent_table)
plt.ylabel('Count', size = 13)
plt.xlabel('Has Questions', size = 13)
plt.title("Existence Of a Questions for Fraudulent", size = 15, weight='bold')
plt.legend(labels=['No questions', 'Has questions'], loc=(0.6, 0.8),  title='Company Logo')
without_hue(questions_Plot2, fraudulent_table['has_questions'])
plt.show()

# ------------------------------------------------------employment_type---------------------------------------------------------
employment_type_Plot = sns.countplot(x='employment_type', hue='employment_type', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Employment Type', size = 13)
plt.title("Employment Type", size = 15, weight='bold')
plt.legend(labels=['No Description', 'Full-time', 'Other', 'Temporary', 'Pull-time',
                   'Contract'], loc=(0.65, 0.5),  title='employment type')
#without_hue(yardPlot, table.hasYard)
plt.show()

# ------------------------------------------------------required_experience---------------------------------------------------------
required_education_Plot = sns.countplot(x='required_experience', hue='required_experience', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Required Experience', size = 10)
plt.title("Required Experience", size = 15, weight='bold')
#plt.legend(labels=['No Description', 'Other', 'Art/Creative', 'Temporary', 'Pull-time', 'Contract'], loc=(0.65, 0.5),  title='Required Experience')
#without_hue(yardPlot, table.hasYard)
plt.show()

# ------------------------------------------------------required_education---------------------------------------------------------
required_education_Plot = sns.countplot(x='required_education', hue='required_education', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Required Education', size = 10)
plt.title("Required Education", size = 15, weight='bold')
#plt.legend(labels=['No Description', 'Other', 'Art/Creative', 'Temporary', 'Pull-time', 'Contract'], loc=(0.65, 0.5),  title='Required Education')
#without_hue(yardPlot, table.hasYard)
plt.show()

# ------------------------------------------------------industry---------------------------------------------------------
text_data1 = table.groupby('fraudulent')
unique_binary_vals = text_data1.groups.keys()
true_fraudulent = []
false_fraudulent = []

for binary_val in unique_binary_vals:
    # Get the indices of the DataFrame rows that have the current binary value
    indices = text_data1.get_group(binary_val).index
    # Get the corresponding string values and append to the appropriate list
    string_values = table.loc[indices, 'industry'].tolist()
    if binary_val == 0:
        true_fraudulent.extend(string_values)
    else:
        false_fraudulent.extend(string_values)

# Tokenize the text data into individual words
tokenized_data_true = [word_tokenize(text) for text in true_fraudulent]
tokenized_data_false = [word_tokenize(text) for text in false_fraudulent]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_data_true = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_true]
filtered_data_false = [[word for word in text if word.lower() not in stop_words and word.lower() not in string.punctuation] for text in tokenized_data_false]

# Calculate the frequency of each word
fdist_true = FreqDist([word for text in filtered_data_true for word in text])
fdist_false = FreqDist([word for text in filtered_data_false for word in text])


# Get the 10 most common words
most_common_company_profile_true = fdist_true.most_common(20)
most_common_company_profile_false = fdist_false.most_common(20)

most_common_company_profile_dict_true = {k: v for k, v in most_common_company_profile_true}
most_common_company_profile_dict_false = {k: v for k, v in most_common_company_profile_false}


# Apply the function to the column and create a new column for the top word counts
count_dict_true = {word: 0 for word in most_common_company_profile_dict_true}
count_dict_false = {word: 0 for word in most_common_company_profile_dict_false}

for index, row in table.iterrows():
    sentence = str(row['industry'])
    words = word_tokenize(sentence)  # calculate how many words
    if row['fraudulent'] == 0:
        for word in most_common_company_profile_dict_true:
            if word in words:
                count_dict_true[word] += 1
    else:
        for word in most_common_company_profile_dict_false:
            if word in words:
                count_dict_false[word] += 1

labels_true = list(count_dict_true.keys())
values_true = list(count_dict_true.values())
labels_false = list(count_dict_false.keys())
values_false = list(count_dict_false.values())

#fraudulent=0
plt.barh(labels_true, values_true, color="green")
# Add text labels to the bars
for i, v in enumerate(values_true):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in industry where fraudulent=0")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

#fraudulent=1
plt.barh(labels_false, values_false, color="red")
# Add text labels to the bars
for i, v in enumerate(values_false):
    plt.text(v + 1, i, str(v), ha='left')
plt.title("20 common words in industry where fraudulent=1")
plt.xlabel("Count")
plt.ylabel("20 common words")
# plt.legend(loc=(1, 1), title='words')
plt.show()

# ------------------------------------------------------function---------------------------------------------------------
function_plot = sns.countplot(x='function', hue='function', dodge=False,  data=table)
plt.ylabel('function', size = 13)
plt.xlabel('Count', size = 10)
plt.title("function", size = 15, weight='bold')
plt.legend(loc=(1, 1),  title='function')
#without_hue(yardPlot, table.hasYard)
plt.show()

function_Plot_real = sns.countplot(x='function', hue='function', dodge=False,  data=no_fraudulent_table)
plt.ylabel('function', size = 13)
plt.xlabel('Count', size = 10)
plt.title("function for No Fraudulent Job", size = 15, weight='bold')
plt.legend(loc=(0.5, 0.05),  title='function')
#without_hue(yardPlot, table.hasYard)
plt.show()

# ------------------------------------------------------fraudulent---------------------------------------------------------
fraudulent_Plot = sns.countplot(x='fraudulent', hue='fraudulent', dodge=False,  data=table)
plt.ylabel('Count', size = 13)
plt.xlabel('Fraudulent', size = 13)
plt.title("Fraudulent", size = 15, weight='bold')
plt.legend(labels=['No Fraudulent', 'Fraudulent'], loc=(0.6, 0.8),  title='Fraudulent')
without_hue(fraudulent_Plot, table['fraudulent'])
plt.show()


# #---------------------------------------------------------connection between variable---------------------------------------------
# correlation heatmap
# select the columns  to analyze
subset_table = table[['job_id', 'company_profile_num_of_words', 'description_num_of_words', 'benefits_num_of_words', 'requirements_num_of_words', 'telecommuting',  'has_company_logo',  'has_questions',  'fraudulent']]
heatCorr = sns.heatmap(subset_table.corr())
heatCorr.set_title('Colored Correlation Matrix')
plt.tight_layout()
plt.show()

# correlation matrix
corMat = pd.DataFrame.corr(subset_table)
#covMat = pd.DataFrame.cov(subset_table)
#corMat_export = pd.DataFrame(corMat)

print(corMat)
#print(covMat)
# export DataFrame to Excel
#corMat_export.to_excel('corMat.xlsx', index=False)

# ------------------------------------------------------ Feature Representation

# ------------------------------------------------------ Feature selection
x = table.drop("fraudulent",axis=1)
# Separate the features and target variable
y = table['fraudulent']
# x is for the features and y is for the fraudulent
# Create an instance of SelectKBest and fit it to the data
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)

# Get the scores and p-values of each feature and sort them in descending order
scores = selector.scores_
p_values = selector.pvalues_
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores, 'p-value': p_values})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# Print the top k features with their scores and p-values
k = 30
top_k_features = feature_scores[['Feature', 'Score', 'p-value']][:k]
print(f'The top {k} features are:')
print(top_k_features)

#### gain ration
mutual_info = mutual_info_classif(X, y)

# Fit a decision tree classifier to the data to calculate the split criterion for each feature
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X, y)
split_criterion = dtc.tree_.compute_feature_importances(normalize=False)

# Calculate the Gain Ratio for each feature
gain_ratio = mutual_info / (split_criterion+1e-6)

# Create a new dataframe with the feature names and their corresponding Gain Ratio values
df_gain_ratio = pd.DataFrame({'Feature': X.columns, 'Gain Ratio': gain_ratio})

# Sort the dataframe by Gain Ratio in descending order
df_gain_ratio_sorted = df_gain_ratio.sort_values('Gain Ratio', ascending=False)

# Print the top 30 features by Gain Ratio and their corresponding scores
print(df_gain_ratio_sorted.head(30))

### Wrappers procedures
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Initialize the RFE method and fit the model
rfe = RFE(model, n_features_to_select=20)
rfe.fit(X_scaled, y)

# Get the feature ranking and scoring
ranking = rfe.ranking_
scoring = rfe.estimator_.coef_[0]

# Create a list of tuples with feature names, ranking, and scores
features_list = list(zip(X.columns, ranking, scoring))

# Sort the features by ranking
features_by_ranking = sorted(zip(X.columns, ranking), key=lambda x: x[1])

# Print the features by ranking
print("\nFeatures by ranking:")
for feature in features_by_ranking:
    print("Feature: %s, Rank: %d" % (feature[0], feature[1]))

