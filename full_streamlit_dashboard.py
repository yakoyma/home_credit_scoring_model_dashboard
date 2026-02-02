"""
===============================================================================
Home Credit Scoring Model Project: Dashboard
===============================================================================
"""
# Standard libraries
import os
import time

# Other libraries
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly_express as px


from joblib import load
from shap import Explainer, maskers, plots
from streamlit_shap import st_shap
from sklearn.neighbors import NearestNeighbors



def load_customers_dataset(file, encoding):
    """This function loads customers data from a CSV file located in the
    same directory as the script.

    Args:
        file (str): the name of the CSV file containing the customers data
        encoding (str): optional encoding type for the CSV file

    Returns:
        dataset (pd.DataFrame): the pandas DataFrame containing loaded data
    """

    folder_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(folder_path, file)
    dataset = pd.read_csv(
        dataset_path, encoding=encoding, encoding_errors='ignore')
    return dataset


def get_preprocessing(X):
    """This function encodes object features, scales numerical features,
    and removes correlated features.

    Args:
        X (pd.DataFrame): input DataFrame

    Returns:
        X_preprocessed (pd.DataFrame): the preprocessed DataFrame
    """

    ENCODER_PATH = 'models/encoder/encoder.joblib'
    SCALER_PATH = 'models/scaler/scaler.joblib'
    SELECTOR_PATH = 'models/selector/selector.joblib'

    # Load encoder
    encoder = load(filename=ENCODER_PATH)

    # Load the scaler
    scaler = load(filename=SCALER_PATH)

    # Load the selector
    selector = load(filename=SELECTOR_PATH)

    X_raw = X.drop(['SK ID CURR', 'CODE GENDER'], axis=1)
    numeric_features = X_raw.select_dtypes(include=['number']).columns.tolist()
    object_features = X_raw.select_dtypes(include=['object']).columns.tolist()

    # Encode object features
    X_encoded = encoder.transform(X=X_raw[object_features])

    # Add encoded features to the numerical dataset
    X_numeric = X_raw[numeric_features]
    for i, feature in enumerate(object_features):
        X_numeric[feature] = X_encoded[:, i]

    # Normalisation
    X_preprocessed = pd.DataFrame(
        data=scaler.transform(X_numeric), columns=list(X_numeric.columns))

    # Get a set of correlated features based on the correlation threshold
    relevant_features = [
        'AMT CREDIT', 'AMT ANNUITY', 'AMT INCOME TOTAL', 'DAYS EMPLOYED',
        'ORGANIZATION TYPE', 'DAYS BIRTH', 'NAME FAMILY STATUS',
        'CNT CHILDREN',
    ]
    X_scaled = np.array(X_preprocessed.drop(relevant_features, axis=1))
    X_selected = selector.transform(X_scaled)
    selected_features = list(np.array(X_preprocessed.drop(
        relevant_features, axis=1).columns)[selector.get_support()])
    features = list()
    features.extend(relevant_features)
    features.extend(selected_features)

    # Remove correlated features
    X_preprocessed = X_preprocessed[features]
    return X_preprocessed


def get_application_status(X_customer, model, credit_score_risk_threshold):
    """This function determines the application status (risk and decision)
    for a customer using a pre-trained Machine Learning model.

    Args:
        X_customer (pd.DataFrame): the customer data
        model: the pre-trained Machine Learning model
        credit_score_risk_threshold (int): the risk threshold of the credit
                                           score for decision

    Returns:
        tuple (str, str, int):
            - status (str): the status of the home credit application
                            (granted or refused)
            - situation (str): the customer profile (at risk or without risk)
            - score (int): the credit score of the customer
    """

    # Make prediction
    y_proba = model.predict_proba(X_customer)[:, 1]

    # Calculate the credit score of the customer
    probability = y_proba[0]
    score = int(probability * 100)

    # Determine the class of the customer
    customer_class = np.where(score > credit_score_risk_threshold, 1, 0)

    # Determine the result of the home credit application
    if customer_class == 1:
        situation = 'At risk'
        status = 'Refused'
    else:
        situation = 'Without risk'
        status = 'Granted'
    return status, situation, score


def customers_description(df):
    """This function creates a Pandas DataFrame with a user-friendly
    description of customers.

    Args:
        df (pd.DataFrame): input Pandas DataFrame with customers data

    Returns:
        df (pd.DataFrame): the Pandas DataFrame with renamed and formatted
                           customer features
    """

    old_relevant_features = [
        'SK ID CURR', 'AMT CREDIT', 'AMT ANNUITY', 'AMT INCOME TOTAL',
        'DAYS EMPLOYED', 'ORGANIZATION TYPE', 'DAYS BIRTH', 'CODE GENDER',
        'NAME FAMILY STATUS', 'CNT CHILDREN'
    ]
    df = df.filter(old_relevant_features)
    df['SK ID CURR'] = list(df['SK ID CURR'])
    df['AMT CREDIT'] = list(df['AMT CREDIT'].astype('int64'))
    df['AMT ANNUITY'] = list(df['AMT ANNUITY'].astype('int64'))
    df['AMT INCOME TOTAL'] = list(df['AMT INCOME TOTAL'].astype('int64'))
    df['DAYS EMPLOYED'] = list(df['DAYS EMPLOYED'].abs().astype('int64'))
    df['ORGANIZATION TYPE'] = list(df['ORGANIZATION TYPE'].astype(str))
    df['AGE'] = df['DAYS BIRTH'] / 365
    df['AGE'] = list(df['AGE'].abs().astype('int64'))
    df['CODE GENDER'] = list(df['CODE GENDER'].astype(str))
    df['NAME FAMILY STATUS'] = list(df['NAME FAMILY STATUS'])
    df['CNT CHILDREN'] = list(df['CNT CHILDREN'].astype('int64'))

    features_mapping = {
        'SK ID CURR': 'Customer ID',
        'AMT CREDIT': 'Credit amount ($)',
        'AMT ANNUITY': 'Loan annuity ($)',
        'AMT INCOME TOTAL': 'Income ($)',
        'DAYS EMPLOYED': 'Days employed',
        'ORGANIZATION TYPE': 'Organization type',
        'AGE': 'Age (years)',
        'CODE GENDER': 'Gender',
        'NAME FAMILY STATUS': 'Family status',
        'CNT CHILDREN': 'Number of children'
    }
    df = df.rename(columns=features_mapping)

    new_relevant_features = [
        'Customer ID', 'Credit amount ($)', 'Loan annuity ($)', 'Income ($)',
        'Days employed', 'Organization type', 'Age (years)', 'Gender',
        'Family status', 'Number of children'
    ]
    df['Application status'] = ''
    df['Situation'] = ''
    df['Score'] = np.nan
    df = df.filter(
        ['Application status', 'Situation', 'Score'] + new_relevant_features)
    return df


def get_similar_customers(X_preprocessed, index, similar_customers_number):
    """This function finds the most similar customers to a given customer
    using the Nearest Neighbors algorithm.

    Args:
        X_preprocessed (pd.DataFrame): the preprocessed DataFrame with data of
                                       customers
        index (int): index of the customer of interest
        similar_customers_number (int): the number of similar customers to find

    Returns:
        index_list (list): indices of the most similar customers
    """

    # Instantiate the model
    nn_model = NearestNeighbors(
        n_neighbors=similar_customers_number + 1, n_jobs=-1)

    # Train the model
    nn_model.fit(X=X_preprocessed)

    # Retrieve indices
    indice = nn_model.kneighbors(X_preprocessed, return_distance=False)
    index_list = next(
        indice_sublist for indice_sublist in indice if index in indice_sublist)
    return index_list


def main():


    try:

        CREDIT_SCORE_BEST_RISK_THRESHOLD_PATH = (
            'models/flaml/results/credit_score_best_risk_threshold.npy'
        )
        MODEL_PATH = 'models/flaml/model.joblib'

        st.set_page_config(layout='wide')
        st.title("HOME CREDIT SCORING DASHBOARD")

        sns.set_style('whitegrid')

        # Load the datasets
        dataset = load_customers_dataset(
            file='dashboard_dataset.csv', encoding='utf-8')
        X_train = load_customers_dataset(
            file='train_dataset.csv', encoding='utf-8')

        # Transform the customers dataset
        X = dataset.copy()
        X.columns = X.columns.str.replace('_', ' ')
        X_preprocessed = get_preprocessing(X)

        # Select the customer of interest
        customers_list = list(X['SK ID CURR'])
        customer_id = st.sidebar.selectbox(
            'Select or enter a customer ID', customers_list)

        # Retrieve the customer's dataset
        customer_index = X.loc[X['SK ID CURR'] == customer_id].index[0]
        customer_dataset = X_preprocessed.loc[[customer_index]]

        # Load the pre-trained ML model
        model = load(filename=MODEL_PATH)

        # Select the risk threshold of the credit score
        credit_score_best_risk_threshold = int(np.load(
            file=CREDIT_SCORE_BEST_RISK_THRESHOLD_PATH))
        risk_threshold = st.sidebar.slider(
            label='Select the risk threshold of the credit score',
            min_value=0,
            max_value=100,
            value=credit_score_best_risk_threshold
        )

        st.header("Application status")
        st.write(
            "The credit score varies between 0 and 100. "
            "According to the model evaluation, the best (default) value of "
            "the risk threshold is {}. That is why, customers with scores "
            "above {} have a risk profile.".format(
                credit_score_best_risk_threshold, risk_threshold)
        )

        # Description of the customer
        customer_description_viz = customers_description(
            X.loc[X['SK ID CURR'] == customer_id])

        # Get application status
        try:
            status, situation, score = get_application_status(
                customer_dataset, model, risk_threshold)
            customer_description_viz['Application status'] = status
            customer_description_viz['Situation'] = situation
            customer_description_viz['Score'] = score

            st.write(
                "**The score of the customer N°{} is {}. The customer's "
                "situation is {}. Therefore, the credit application status "
                "is {}.**".format(
                    customer_id, score, situation.lower(), status.lower())
            )
            st.header("Descriptive information of the customer")
            st.dataframe(customer_description_viz.set_index('Customer ID'))
        except Exception as error:
            st.write(
                "This result is not available. "
                f"The following error occurred: {error}"
            )

        # Information of the customer
        display_information = st.sidebar.selectbox(
            "Select the topic to display",
            [
                "Visualisations",
                "Similar customers",
                "Global interpretability of the model",
                "Local interpretability of the model",
                "Customer data"
            ]
        )

        # Select the number of similar customers
        similar_customers_number = st.sidebar.slider(
            label='Select the number of similar customers',
            min_value=0,
            max_value=int(X.shape[0] * 0.05),
            value=10
        )
        with st.spinner("Loading..."):
            similar_customers_list = get_similar_customers(
                X_preprocessed, customer_index, similar_customers_number)
        similar_customers_dataset = X.loc[similar_customers_list]
        info_viz = customers_description(similar_customers_dataset)
        similar_customers_viz = info_viz.drop([customer_index])

        if display_information == "Visualisations":
            st.header("Visualisations of the descriptive information")
            st.write(
                "Visualisations allow you to compare the descriptive "
                "information of the customer N°{} with the {} similar "
                "clients.".format(customer_id, similar_customers_number)
            )
            display_description = st.sidebar.selectbox(
                "Select the descriptive information",
                [
                    "Financial information", "Gender", "Age",
                    "Professional status", "Number of children",
                    "Family status", "Business segment"
                ]
            )
            if display_description == "Financial information":
                try:
                    st.bar_chart(
                        data=info_viz.filter(
                            [
                                'Customer ID', 'Credit amount ($)',
                                'Loan annuity ($)', 'Income ($)'
                            ]
                        ),
                        x='Customer ID',
                        y=[
                            'Credit amount ($)',
                            'Loan annuity ($)',
                            'Income ($)'
                        ],
                        sort=True,
                        stack=True
                    )
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
                financial_information_features = st.multiselect(
                    label="Select the financial information to be displayed:",
                    options=[
                        'Credit amount ($)', 'Loan annuity ($)', 'Income ($)']
                )
                time.sleep(5)
                try:
                    if not financial_information_features:
                        st.info("Sorry, no information is given!")
                    else:
                        st.dataframe(info_viz.set_index(
                            'Customer ID')[financial_information_features])
                        ax = info_viz.set_index('Customer ID')[
                            financial_information_features].plot(kind='bar')
                        fig = ax.get_figure()
                        st.pyplot(fig)
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
            elif display_description == "Gender":
                st.subheader("Gender")
                try:
                    gender_viz = info_viz.groupby('Gender').count()
                    gender_viz.reset_index(inplace=True, drop=False)
                    gender_viz = gender_viz.rename(
                        columns={'Customer ID': 'Count'}).sort_values(
                        by=['Count'], ascending=False)
                    fig = px.bar(
                        data_frame=gender_viz,
                        x='Gender',
                        y='Count',
                        height=300
                    )
                    st.write(fig)
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
            elif display_description == "Age":
                st.subheader("Age (years)")
                try:
                    st.bar_chart(
                        data=info_viz.filter(['Customer ID', 'Age (years)']),
                        x='Customer ID',
                        y='Age (years)',
                        sort='-Age (years)'
                    )
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
            elif display_description == "Professional status":
                st.subheader("Professional status")
                st.write(
                    "You could see below the number of days elapsed since "
                    "the beginning of the last employment contract."
                )
                try:
                    st.bar_chart(
                        data=info_viz.filter(['Customer ID', 'Days employed']),
                        x='Customer ID',
                        y='Days employed',
                        sort='-Days employed'
                    )
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
            elif display_description == "Number of children":
                st.subheader("Number of children")
                try:
                    st.bar_chart(
                        data=info_viz.filter(
                            ['Customer ID', 'Number of children']),
                        x='Customer ID',
                        y='Number of children',
                        sort='-Number of children'
                    )
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
            elif display_description == "Family status":
                st.subheader("Marital status")
                try:
                    family_status_viz = info_viz.groupby(
                        'Family status').count()
                    family_status_viz.reset_index(inplace=True, drop=False)
                    family_status_viz = family_status_viz.rename(
                        columns={'Customer ID': 'Count'}).sort_values(
                        by=['Count'], ascending=False)
                    fig = px.bar(
                        data_frame=family_status_viz,
                        x='Family status',
                        y='Count',
                        height=300
                    )
                    st.write(fig)
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
            elif display_description == "Business segment":
                st.subheader("Business segment")
                try:
                    business_segment_viz = info_viz.groupby(
                        'Organization type').count()
                    business_segment_viz.reset_index(inplace=True, drop=False)
                    business_segment_viz = business_segment_viz.rename(
                        columns={'Customer ID': 'Count'}).sort_values(
                        by=['Count'], ascending=False)
                    fig = px.bar(
                        data_frame=business_segment_viz,
                        x='Organization type',
                        y='Count',
                        height=500
                    )
                    st.write(fig)
                except Exception as error:
                    st.write(
                        "This chart is not available. "
                        f"The following error occurred: {error}"
                    )
        elif display_information == "Similar customers":
            st.header("Similar customers")
            st.write(
                "The grouping of similar customers allows you to compare "
                "the customer N°{} with {} similar customers.".format(
                customer_id, similar_customers_number)
            )
            try:
                score_list, situation_list, status_list = [], [], []
                for id in list(similar_customers_viz['Customer ID']):
                    similar_customer_index = X.loc[X[
                        'SK ID CURR'] == id].index[0]
                    similar_customer_dataset = X_preprocessed.loc[
                        [similar_customer_index]]
                    status, situation, score = get_application_status(
                        similar_customer_dataset, model, risk_threshold)
                    score_list.append(score)
                    situation_list.append(situation)
                    status_list.append(status)
                similar_customers_viz['Application status'] = status_list
                similar_customers_viz['Situation'] = situation_list
                similar_customers_viz['Score'] = score_list
                st.dataframe(similar_customers_viz.set_index('Customer ID'))
            except Exception as error:
                st.write(
                    "This result is not available. "
                    f"The following error occurred: {error}"
                )
        elif display_information == "Global interpretability of the model":
            st.header("Global interpretability of the model")
            st.write(
                "The global interpretability provides overall comprehension "
                "of relevant features for the model prediction."
            )
            try:
                explainer = Explainer(
                    model=model,
                    masker=maskers.Independent(
                        data=X_train.drop(['LABEL'], axis=1), max_samples=1000)
                )
                shap_values = explainer(X_preprocessed)
                max_features = st.sidebar.slider(
                    label='Select the number of columns',
                    min_value=0,
                    max_value=int(X_preprocessed.shape[1] * 0.8),
                    value=int(X_preprocessed.shape[1] * 0.4)
                )
                st_shap(plots.beeswarm(
                    shap_values=shap_values, max_display=max_features))
            except Exception as error:
                st.write(
                    "This chart is not available. "
                    f"The following error occurred: {error}"
                )
            st.write(
                "The GDPR (Article 22) provides restrictive rules to prevent "
                "human from being subjected to automated decisions that have "
                "legal or significant effects."
            )
        elif display_information == "Local interpretability of the model":
            st.header("Local interpretability of the model")
            st.write(
                "The local interpretability allows you to determine "
                "the effects of features in the result of predicting "
                "the customer's N°{} score.".format(customer_id)
            )
            try:
                explainer = Explainer(
                    model=model,
                    masker=maskers.Independent(
                        data=X_train.drop(['LABEL'], axis=1), max_samples=1000)
                )
                shap_values = explainer(X_preprocessed)
                max_features = st.sidebar.slider(
                    label='Select the number of columns',
                    min_value=0,
                    max_value=int(X_preprocessed.shape[1] * 0.8),
                    value=int(X_preprocessed.shape[1] * 0.4)
                )
                st_shap(
                    plots.waterfall(
                        shap_values=shap_values[customer_index],
                        max_display=max_features
                    )
                )
            except Exception as error:
                st.write(
                    "This chart is not available. "
                    f"The following error occurred: {error}"
                )
            st.write(
                "The GDPR (Article 22) provides restrictive rules to prevent "
                "human from being subjected to automated decisions that have "
                "legal or significant effects."
            )
        elif display_information == "Customer data":
            st.header("Customer data")
            st.write("Display data of the customer.")
            customer_data = dataset.loc[dataset['SK_ID_CURR'] == customer_id]
            customer_data['DATA'] = 'Data'
            customer_data.set_index('DATA', inplace=True)
            st.dataframe(customer_data.transpose())

            # Load the dataset description
            st.subheader("Description of the dataset")
            st.write("Display the description of the customer's dataset.")
            dataset_description = load_customers_dataset(
                file='HomeCredit_columns_description.csv', encoding='utf-8')
            st.dataframe(dataset_description.set_index('Row'))

    except Exception as error:
        st.write(
            "This dashboard is not available. "
            f"The following error occurred: {error}"
        )



if __name__ == '__main__':
    main()
