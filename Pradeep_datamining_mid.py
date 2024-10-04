#!/usr/bin/env python
# coding: utf-8

# ## Brute Force Approach for Transaction Analysis

# In[1]:


import pandas as pd
import itertools
import time

def load_transactions_from_csv(file_path):
    data = pd.read_csv(file_path, usecols=[1], skiprows=1, names=['items'])
    return data['items'].apply(lambda x: set(x.split(' , '))).tolist()

def get_support(itemset, transactions):
    return sum(1 for transaction in transactions if set(itemset).issubset(transaction))

def generate_frequent_itemsets(transactions, min_support, max_len=3):
    all_items = {item for transaction in transactions for item in transaction}
    frequent_itemsets = []
    
    for size in range(1, max_len + 1):
        for combination in itertools.combinations(all_items, size):
            support_count = get_support(combination, transactions)
            if support_count >= min_support:
                frequent_itemsets.append((set(combination), support_count))
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    
    for itemset, support_count in frequent_itemsets:
        for subset in map(set, itertools.chain.from_iterable(itertools.combinations(itemset, r) for r in range(1, len(itemset)))):
            remainder = itemset - subset
            if not remainder:
                continue
            subset_support = get_support(subset, transactions)
            confidence = support_count / subset_support if subset_support else 0
            
            if confidence >= min_confidence:
                rules.append((subset, remainder, confidence))
    
    return rules

def analyze_transactions(file_path, min_support=0.5, min_confidence=0.5):
    transactions = load_transactions_from_csv(file_path)
    frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
    association_rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
    
    return frequent_itemsets, association_rules

start_time = time.time()

csv_files = [
    '/Users/pradeepvarma/Downloads/dataset/bestbuy_transactions.csv',
    '/Users/pradeepvarma/Downloads/dataset/nike_transactions.csv',
    '/Users/pradeepvarma/Downloads/dataset/kmart_transactions.csv',
    '/Users/pradeepvarma/Downloads/dataset/costco_transactions.csv',
    '/Users/pradeepvarma/Downloads/dataset/amazon_transactions.csv'
]

for csv_file in csv_files:
    itemsets, rules = analyze_transactions(csv_file, min_support=0.5, min_confidence=0.5)
    print(f"Results for {csv_file}:")
    print("Frequent Itemsets:", itemsets)
    print("Association Rules:", rules)
    print("\n")

print("Total Execution Time:", time.time() - start_time)


# In[2]:


import pandas as pd
from time import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def load_transaction_data(csv_file):
    transaction_list = pd.read_csv(csv_file)['Transaction'].str.split(', ').tolist()
    return transaction_list

def analyze_transactions(dataset_path, support_threshold, confidence_level):
    start_time = time()
    
    transactions = load_transaction_data(dataset_path)
    
    encoder = TransactionEncoder()
    transaction_matrix = encoder.fit_transform(transactions)
    transaction_df = pd.DataFrame(transaction_matrix, columns=encoder.columns_)
    
    frequent_itemsets = apriori(transaction_df, min_support=support_threshold, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_level)
    
    dataset_name = dataset_path.split('/')[-1] 
    print(f"Transactions in {dataset_name}:")
    for transaction in transactions:
        print(transaction)
    
    print("\nGenerated Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    
    end_time = time()
    print(f"\nTime taken for {dataset_name}: {end_time - start_time:.2f} seconds\n")

def start_analysis():
    datasets = [
        '/Users/pradeepvarma/Downloads/dataset/bestbuy_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/nike_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/kmart_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/costco_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/amazon_transactions.csv'
    ]
    
    support_threshold = float(input("Enter the minimum support value (e.g., 0.05 for 5%): "))
    confidence_level = float(input("Enter the minimum confidence value (e.g., 0.5 for 50%): "))

    for file_path in datasets:
        analyze_transactions(file_path, support_threshold, confidence_level)

if __name__ == "__main__":
    start_analysis()


# ## Apriori Algorithm for Transaction Analysis

# In[3]:


import pandas as pd
from time import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def load_and_filter_transactions(file_path, min_items=2):
    df = pd.read_csv(file_path)
    
    transactions = df['Transaction'].apply(lambda x: x.split(', '))
    filtered_transactions = [tx for tx in transactions if len(tx) >= min_items]
    return filtered_transactions

def perform_transaction_analysis(file_path, min_support, min_confidence):
    start_time = time()  

    transactions = load_and_filter_transactions(file_path)
    
    encoder = TransactionEncoder()
    transaction_array = encoder.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(transaction_array, columns=encoder.columns_)
    
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    print(f"Filtered Transactions from {file_path.split('/')[-1]}:")
    for transaction in transactions:
        print(transaction)
        
    print("\nGenerated Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    
    elapsed_time = time() - start_time
    print(f"\nTime taken for {file_path.split('/')[-1]}: {elapsed_time:.2f} seconds\n")

def execute_analysis():
    datasets = {
        1: '/Users/pradeepvarma/Downloads/dataset/amazon_transactions.csv',
        2: '/Users/pradeepvarma/Downloads/dataset/costco_transactions.csv',
        3: '/Users/pradeepvarma/Downloads/dataset/bestbuy_transactions.csv',
        4: '/Users/pradeepvarma/Downloads/dataset/nike_transactions.csv',
        5: '/Users/pradeepvarma/Downloads/dataset/kmart_transactions.csv',
    }

    print("Please select dataset(s):")
    for key, value in datasets.items():
        print(f"{key} - {value.split('/')[-1]}")

    selected_datasets = input("Enter your choices (e.g., 1,3): ").split(',')
    min_support = float(input("Enter minimum support value (e.g., 0.05): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5): "))

    for dataset in selected_datasets:
        perform_transaction_analysis(datasets[int(dataset)], min_support, min_confidence)

if __name__ == "__main__":
    execute_analysis()


# ## FP-Growth Algorithm for Transaction Analysis

# In[5]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def load_transactions_from_csv(file_path):
    return pd.read_csv(file_path)['Transaction'].str.split(', ').tolist()

def perform_fp_growth_analysis(file_path, support_threshold, confidence_threshold):
    transactions = load_transactions_from_csv(file_path)
    
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.columns_)
    
    frequent_itemsets = fpgrowth(encoded_df, min_support=support_threshold, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)
    
    print(f"Frequent Itemsets from {file_path.split('/')[-1]}:")
    print(frequent_itemsets)
    print("\nDerived Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    print("\n")

def start_analysis():
    dataset_file_paths = [
        '/Users/pradeepvarma/Downloads/dataset/bestbuy_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/nike_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/kmart_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/costco_transactions.csv',
        '/Users/pradeepvarma/Downloads/dataset/amazon_transactions.csv'
    ]
    
    min_support = float(input("Enter the minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter the minimum confidence value (e.g., 0.5 for 50%): "))
    
    for file_path in dataset_file_paths:
        perform_fp_growth_analysis(file_path, min_support, min_confidence)

if __name__ == "__main__":
    start_analysis()


# In[6]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import time

def load_transactions(file_path):
    transactions = pd.read_csv(file_path)['Transaction'].str.split(', ').tolist()
    return transactions

def perform_analysis(file_path, min_support, min_confidence):
    transactions = load_transactions(file_path)
    
    encoder = TransactionEncoder()
    transformed_data = encoder.fit(transactions).transform(transactions)
    transactions_df = pd.DataFrame(transformed_data, columns=encoder.columns_)
    
    frequent_itemsets = fpgrowth(transactions_df, min_support=min_support, use_colnames=True)
    
    association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    print(f"\nFrequent Itemsets from {file_path.split('/')[-1]}:")
    print(frequent_itemsets)
    print("\nDerived Association Rules:")
    print(association_rules_df[['antecedents', 'consequents', 'support', 'confidence']])
    print("\n")

def execute_analysis():
    store_names = ['Amazon', 'Costco', 'Best Buy', 'Walgreens', 'Walmart']
    dataset_paths = {
        '1': '/Users/pradeepvarma/Downloads/dataset/amazon_transactions.csv',
        '2': '/Users/pradeepvarma/Downloads/dataset/costco_transactions.csv',
        '3': '/Users/pradeepvarma/Downloads/dataset/bestbuy_transactions.csv',
        '4': '/Users/pradeepvarma/Downloads/dataset/nike_transactions.csv',
        '5': '/Users/pradeepvarma/Downloads/dataset/kmart_transactions.csv',
    }
    
    print("Select dataset(s) for analysis:")
    for index, name in enumerate(store_names, start=1):
        print(f"{index} - {name}")
    user_choices = input("Enter dataset numbers (space-separated, e.g., 1 3): ").split()
    
    min_support_value = float(input("\nMinimum support (e.g., 0.05): "))
    min_confidence_value = float(input("Minimum confidence (e.g., 0.5): "))
    
    for choice in user_choices:
        if choice in dataset_paths:
            perform_analysis(dataset_paths[choice], min_support_value, min_confidence_value)
        else:
            print(f"Invalid choice: {choice}")

if __name__ == "__main__":
    execute_analysis()


# In[ ]:




