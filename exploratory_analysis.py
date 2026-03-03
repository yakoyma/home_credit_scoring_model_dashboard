"""
===============================================================================
Home Credit Scoring Model Project: Exploratory Analysis
===============================================================================

This file is organised as follows:
1. Load and explore raw datasets
2. Selection of final datasets
3. Cleanse and save datasets
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import pandas as pd
import sweetviz as sv
import ydata_profiling


from sweetviz import analyze
from ydata_profiling import ProfileReport
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))



# Constants
MAX_ROWS_DISPLAY = 100
MAX_COLUMNS_DISPLAY = 150

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load and explore raw datasets
===============================================================================
"""
print(f'\n\n\n1. Load and explore raw datasets')

# Paths of datasets
INPUT_CSV_1 = 'datasets/bureau.csv'
INPUT_CSV_2 = 'datasets/bureau_balance.csv'
INPUT_CSV_3 = 'datasets/credit_card_balance.csv'
INPUT_CSV_4 = 'datasets/HomeCredit_columns_description.csv'
INPUT_CSV_5 = 'datasets/installments_payments.csv'
INPUT_CSV_6 = 'datasets/POS_CASH_balance.csv'
INPUT_CSV_7 = 'datasets/previous_application.csv'
INPUT_CSV_8 = 'datasets/sample_submission.csv'
INPUT_CSV_9 = 'datasets/application_train.csv'
INPUT_CSV_10 = 'datasets/application_test.csv'


# Load bureau dataset
print(f'\n\nLoad bureau dataset:')
bureau = load_dataset(INPUT_CSV_1, 'utf-8')

# Display dataset information and description
dataset_info_description(bureau, 15)

# Generate dataset report
bureau_report_sv = analyze(source=bureau)
bureau_report_sv.show_html('datasets/bureau_report_sv.html')
bureau_report_ydp = ProfileReport(df=bureau, title='Bureau Dataset Report')
bureau_report_ydp.to_file('datasets/bureau_report_ydp.html')


# Load bureau balance dataset
print(f'\n\nLoad bureau balance dataset:')
bureau_balance = load_dataset(INPUT_CSV_2, 'utf-8')

# Display dataset information and description
dataset_info_description(bureau_balance, 15)

# Generate dataset report
bureau_balance_report_sv = analyze(source=bureau_balance)
bureau_balance_report_sv.show_html('datasets/bureau_balance_report_sv.html')
bureau_balance_report_ydp = ProfileReport(
    df=bureau_balance, title='Bureau Balance Dataset Report')
bureau_balance_report_ydp.to_file('datasets/bureau_balance_report_ydp.html')


# Load credit card balance dataset
print(f'\n\nLoad credit card balance dataset:')
credit_card_balance = load_dataset(INPUT_CSV_3, 'utf-8')

# Display dataset information and description
dataset_info_description(credit_card_balance, 15)

# Generate dataset report
credit_card_balance_report_sv = analyze(source=credit_card_balance)
credit_card_balance_report_sv.show_html(
    'datasets/credit_card_balance_report_sv.html')
credit_card_balance_report_ydp = ProfileReport(
    df=credit_card_balance, title='Credit Card Balance Dataset Report')
credit_card_balance_report_ydp.to_file(
    'datasets/credit_card_balance_report_ydp.html')


# Load homecredit columns description dataset
print(f'\n\nLoad homecredit columns description dataset:')
homecredit_columns_description = load_dataset(INPUT_CSV_4, 'utf-8')

# Display dataset information and description
dataset_info_description(homecredit_columns_description, 15)

# Generate dataset report
homecredit_columns_description_report_sv = analyze(
    source=homecredit_columns_description)
homecredit_columns_description_report_sv.show_html(
    'datasets/homecredit_columns_description_report_sv.html')
homecredit_columns_description_report_ydp = ProfileReport(
    df=homecredit_columns_description,
    title='HomeCredit Columns Description Dataset Report'
)
homecredit_columns_description_report_ydp.to_file(
    'datasets/homecredit_columns_description_report_ydp.html')


# Load installments payments dataset
print(f'\n\nLoad installments payments dataset:')
installments_payments = load_dataset(INPUT_CSV_5, 'utf-8')

# Display dataset information and description
dataset_info_description(installments_payments, 15)

# Generate dataset report
installments_payments_report_sv = analyze(source=installments_payments)
installments_payments_report_sv.show_html(
    'datasets/installments_payments_report_sv.html')
installments_payments_report_ydp = ProfileReport(
    df=installments_payments, title='Installments Payments Dataset Report')
installments_payments_report_ydp.to_file(
    'datasets/installments_payments_report_ydp.html')


# Load pos cash balance dataset
print(f'\n\nLoad pos cash balance dataset:')
pos_cash_balance = load_dataset(INPUT_CSV_6, 'utf-8')

# Display dataset information and description
dataset_info_description(pos_cash_balance, 15)

# Generate dataset report
pos_cash_balance_report_sv = analyze(source=pos_cash_balance)
pos_cash_balance_report_sv.show_html(
    'datasets/pos_cash_balance_report_sv.html')
pos_cash_balance_report_ydp = ProfileReport(
    df=pos_cash_balance, title='Posh Cash Balance Dataset Report')
pos_cash_balance_report_ydp.to_file(
    'datasets/pos_cash_balance_report_ydp.html')


# Load previous application dataset
print(f'\n\nLoad previous application dataset:')
previous_application = load_dataset(INPUT_CSV_7, 'utf-8')

# Display dataset information and description
dataset_info_description(previous_application, 15)

# Generate dataset report
previous_application_report_sv = analyze(source=previous_application)
previous_application_report_sv.show_html(
    'datasets/previous_application_report_sv.html')
previous_application_report_ydp = ProfileReport(
    df=previous_application, title='Previous Application Dataset Report')
previous_application_report_ydp.to_file(
    'datasets/previous_application_report_ydp.html')


# Load sample submission dataset
print(f'\n\nLoad sample submission dataset:')
sample_submission = load_dataset(INPUT_CSV_8, 'utf-8')

# Display dataset information and description
dataset_info_description(sample_submission, 15)

# Generate dataset report
sample_submission_report_sv = analyze(source=sample_submission)
sample_submission_report_sv.show_html(
    'datasets/sample_submission_report_sv.html')
sample_submission_report_ydp = ProfileReport(
    df=sample_submission, title='Sample Submission Dataset Report')
sample_submission_report_ydp.to_file(
    'datasets/sample_submission_report_ydp.html')


# Load application train dataset
print(f'\n\nLoad application train dataset:')
application_train = load_dataset(INPUT_CSV_9, 'utf-8')

# Display dataset information and description
dataset_info_description(application_train, 15)

# Generate dataset report
application_train_report_sv = analyze(source=application_train)
application_train_report_sv.show_html(
    'datasets/application_train_report_sv.html')
application_train_report_ydp = ProfileReport(
    df=application_train, title='Application Train Dataset Report')
application_train_report_ydp.to_file(
    'datasets/application_train_report_ydp.html')


# Load application test dataset
print(f'\n\nLoad application test dataset:')
application_test = load_dataset(INPUT_CSV_10, 'utf-8')

# Display dataset information and description
dataset_info_description(application_test, 15)

# Generate dataset report
application_test_report_sv = analyze(source=application_test)
application_test_report_sv.show_html(
    'datasets/application_test_report_sv.html')
application_test_report_ydp = ProfileReport(
    df=application_test, title='Application Test Dataset Report')
application_test_report_ydp.to_file(
    'datasets/application_test_report_ydp.html')



"""
===============================================================================
2. Selection of final datasets
===============================================================================
"""
print(f'\n\n\n2. Selection of final datasets')
customers_dataset = application_train.copy()
dashboard_dataset = application_test.copy()

# Create new relevant features
customers_dataset['INCOME_BY_PERSON'] = customers_dataset[
    'AMT_INCOME_TOTAL'] / customers_dataset['CNT_FAM_MEMBERS']
customers_dataset['ANNUITY_INCOME_RATE'] = customers_dataset[
    'AMT_ANNUITY'] / customers_dataset['AMT_INCOME_TOTAL']
customers_dataset['DAYS_EMPLOYED_RATE'] = customers_dataset[
    'DAYS_EMPLOYED'] / customers_dataset['DAYS_BIRTH']
customers_dataset['INCOME_CREDIT_RATE'] = customers_dataset[
    'AMT_INCOME_TOTAL'] / customers_dataset['AMT_CREDIT']
customers_dataset['INCOME_PART_RATE'] = customers_dataset[
    'INCOME_BY_PERSON'] / customers_dataset['AMT_INCOME_TOTAL']
customers_dataset['PAYMENT_RATE'] = customers_dataset[
    'AMT_ANNUITY'] / customers_dataset['AMT_CREDIT']

# Display customers dataset information and description
dataset_info_description(customers_dataset, 15)


dashboard_dataset['INCOME_BY_PERSON'] = dashboard_dataset[
    'AMT_INCOME_TOTAL'] / dashboard_dataset['CNT_FAM_MEMBERS']
dashboard_dataset['ANNUITY_INCOME_RATE'] = dashboard_dataset[
    'AMT_ANNUITY'] / dashboard_dataset['AMT_INCOME_TOTAL']
dashboard_dataset['DAYS_EMPLOYED_RATE'] = dashboard_dataset[
    'DAYS_EMPLOYED'] / dashboard_dataset['DAYS_BIRTH']
dashboard_dataset['INCOME_CREDIT_RATE'] = dashboard_dataset[
    'AMT_INCOME_TOTAL'] / dashboard_dataset['AMT_CREDIT']
dashboard_dataset['INCOME_PART_RATE'] = dashboard_dataset[
    'INCOME_BY_PERSON'] / dashboard_dataset['AMT_INCOME_TOTAL']
dashboard_dataset['PAYMENT_RATE'] = dashboard_dataset[
    'AMT_ANNUITY'] / dashboard_dataset['AMT_CREDIT']

# Display dashboard dataset information and description
dataset_info_description(dashboard_dataset, 15)



"""
===============================================================================
3. Cleanse and save datasets
===============================================================================
"""
print(f'\n\n\n3. Cleanse and save datasets')

# Management of missing data
print('\n\nManagement of missing data:')
customers_dataset = customers_dataset.dropna()
customers_dataset.reset_index(inplace=True, drop=True)
dashboard_dataset = dashboard_dataset.dropna()
dashboard_dataset.reset_index(inplace=True, drop=True)


# Management of duplicates
print('\n\nManagement of duplicates:')
customers_dataset_duplicate = customers_dataset[customers_dataset.duplicated()]
print('\nDimensions of customers dataset duplicate: {}'.format(
    customers_dataset_duplicate))

# Display customers dataset information and description
dataset_info_description(customers_dataset, 15)

dashboard_dataset_duplicate = dashboard_dataset[dashboard_dataset.duplicated()]
print('\nDimensions of dashboard dataset duplicate: {}'.format(
    dashboard_dataset_duplicate))

# Display dashboard dataset information and description
dataset_info_description(dashboard_dataset, 15)


# Generate customers dataset report
customers_dataset_report_sv = analyze(source=customers_dataset)
customers_dataset_report_sv.show_html(
    'datasets/customers_dataset_report_sv.html')
customers_dataset_report_ydp = ProfileReport(
    df=customers_dataset, title='Customers Dataset Report')
customers_dataset_report_ydp.to_file(
    'datasets/customers_dataset_report_ydp.html')


# Generate dashboard dataset report
dashboard_dataset_report_sv = analyze(source=dashboard_dataset)
dashboard_dataset_report_sv.show_html(
    'datasets/dashboard_dataset_report_sv.html')
dashboard_dataset_report_ydp = ProfileReport(
    df=dashboard_dataset, title='Dataset Report')
dashboard_dataset_report_ydp.to_file(
    'datasets/dashboard_dataset_report_ydp.html')


# Save customers dataset in CSV format
customers_dataset.to_csv('datasets/customers_dataset.csv', index=False)

# Save dashboard dataset in CSV format
dashboard_dataset.to_csv('datasets/dashboard_dataset.csv', index=False)
