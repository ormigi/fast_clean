#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning Easy
# ##### #5 DIY Python Functions for Data Cleaning, by Matthew Mayo on July 22, 2024 in Resources
# ##### #Further data cleaning, by Mirela Giantaru, 02-AUG-2024
# 

# In[2]:


import re
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Union, Optional


# ### 1. Remove Multiple Spaces
# ##### Our first DIY function is meant to remove excessive whitespace from text. If we want neither multiple spaces within a string, nor excessive leading or trailing spaces, this single line function will take care of it for us. We make use of regular expressions for internal spaces, as well as strip() for trailing/leading whitespace.

# In[3]:


def clean_spaces(text: str) -> str:
    """
    Remove multiple spaces from a string and trim leading/trailing spaces.

    :param text: The input string to clean
    :returns: A string with multiple spaces removed and trimmed
    """
    return re.sub(' +', ' ', str(text).strip())


# In[4]:


messy_text = "This        has   too        many    spaces"   ## testing the function
clean_text = clean_spaces(messy_text)
print(clean_text)


# ###  2. Standardize Date Formats
# #### For datasets with dates running the gamut of internationally acceptable formats? This function will standardize them to a specified format (YYYY-MM-DD).

# In[6]:


def standardize_date(date_string: str) -> Optional[str]:
    """
    Convert various date formats to YYYY-MM-DD.

    :param date_string: The input date string to standardize
    :returns: A standardized date string in YYYY-MM-DD format, or None if parsing fails
    """
    date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Return None if no format matches
    return None


# In[7]:


dates = ["2023-04-01", "01-04-2023", "04/01/2023", "April 1, 2023"]
standardized_dates = [standardize_date(date) for date in dates]
print(standardized_dates)


# ### 3. Handling Missing Values
# ### A way to deal with missing values. 
# ###  numeric data strategy to use (‘mean’, ‘median’, or ‘mode’), 
# ### as well as categorical data strategy (‘mode’ or ‘dummy’).

# In[8]:


def handle_missing(df: pd.DataFrame, numeric_strategy: str = 'mean', categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Fill missing values in a DataFrame.

    :param df: The input DataFrame
    :param numeric_strategy: Strategy for handling missing numeric values ('mean', 'median', or 'mode')
    :param categorical_strategy: Strategy for handling missing categorical values ('mode' or 'dummy')
    :returns: A DataFrame with missing values filled
    """
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            if numeric_strategy == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif numeric_strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            if categorical_strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif categorical_strategy == 'dummy':
                df[column].fillna('Unknown', inplace=True)
    return df


# In[9]:


df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': ['x', 'y', np.nan, 'z']})
cleaned_df = handle_missing(df)
print(cleaned_df)


# ### 4. Remove Outliers
# ### Outliers removal ---  IQR method for removing outliers from the data. 
# ### what dataframe + specify the columns to check for outliers.

# In[12]:


import pandas as pd
import numpy as np
from typing import List

def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the Interquartile Range (IQR) method.

    :param df: The input DataFrame
    :param columns: List of column names to check for outliers
    :param factor: The IQR factor to use (default is 1.5)
    :returns: A DataFrame with outliers removed
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    
    cleaned_df = df[mask]
    
    return cleaned_df


# In[13]:


df = pd.DataFrame({'A': [1, 2, 3, 100, 4, 5], 'B': [10, 20, 30, 40, 50, 1000]})
print("Original DataFrame:")
print(df)
print("\nCleaned DataFrame:")
cleaned_df = remove_outliers_iqr(df, ['A', 'B'])
print(cleaned_df)


# ### 5. Normalize Text Data   ----- or converting all text to lowercase, strip of whitespace, and remove special characters 
# 

# In[15]:


def normalize_text(text: str) -> str:
    """
    Normalize text data by converting to lowercase, removing special characters, and extra spaces.

    :param text: The input text to normalize
    :returns: Normalized text
    """
    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# In[16]:


messy_text = "This is MESSY!!! Text   with $pecial ch@racters."
clean_text = normalize_text(messy_text)
print(clean_text)


# In[18]:


import re

def normalize_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = ' '.join(text.split())
    return text


# In[19]:



messy_text = "This is MESSY!!! Text   with $pecial ch@racters."
clean_text = normalize_text(messy_text)
print(clean_text)


# In[ ]:


##### want to further customize the text normalization process? depending on your specific needs? Here are a few examples:

### Removing Numbers: If you want to remove all digits from the text.
### Stemming or Lemmatization: Reducing words to their base or root form.
### Removing Stop Words: Removing common words like "and", "the", etc., which might not be useful in text analysis.
### Handling Accents and Diacritics: Removing or converting accented characters to their base form.
### Expanding Contractions: Converting contractions like "can't" to "cannot".


# ### Removing Numbers:  to remove all digits from the text.

# def normalize_text_no_numbers(text):
#     # Remove special characters and punctuation
#     text = re.sub(r'[^A-Za-z\s]', '', text)
#     # Convert to lowercase
#     text = text.lower()
#     # Remove extra spaces
#     text = ' '.join(text.split())
#     return text
# 
# messy_text = "This is MESSY!!! Text with $pecial ch@racters and numbers 1234."
# clean_text = normalize_text_no_numbers(messy_text)
# print(clean_text)

# ### Stemming or Lemmatization  -   Using the nltk library for stemming or lemmatization:

# In[21]:


import re
from nltk.stem import PorterStemmer

def normalize_text_stemming(text):
    ps = PorterStemmer()
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

messy_text = "This is MESSY!!! Text with $pecial ch@racters and stemming running runners."
clean_text = normalize_text_stemming(messy_text)
print(clean_text)


# ### Removing Stop Words   Using the nltk library to remove stop words:

# In[ ]:


import re
from nltk.corpus import stopwords

def normalize_text_no_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

messy_text = "This is MESSY!!! Text with $pecial ch@racters and stop words."
clean_text = normalize_text_no_stopwords(messy_text)
print(clean_text)


# ### Handling Accents and Diacritics
# ### Using the unidecode library to handle accents and diacritics:

# In[ ]:


import re
from unidecode import unidecode

def normalize_text_accents(text):
    text = unidecode(text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

messy_text = "Thís ís MÈSSY!!! Têxt with àccented ch@racters."
clean_text = normalize_text_accents(messy_text)
print(clean_text)


# ### Expanding Contractions   # Using the contractions library to expand contractions:

# In[ ]:


import re
import contractions   

def normalize_text_expand_contractions(text):
    text = contractions.fix(text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

messy_text = "This is MESSY!!! Text with $pecial ch@racters and can't contractions."
clean_text = normalize_text_expand_contractions(messy_text)
print(clean_text)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




