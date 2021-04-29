from pycaret.classification import load_model, predict_model
from pycaret.utils import check_metric
import streamlit as st
import pandas as pd 
import numpy as np 
import base64

#load the model
model = load_model('employees_churn_model')

#define prediction fuction

def predict(model , input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    Image.open('./image/download.png').convert('RGB').save('logo.png')
    image = Image.open('logo.png')

    Image.open('employee-turnover.png').convert('RGB').save('employee-turnover2.png')
    image_churn = Image.open('employee-turnover2.png')

    

    add_selectbox = st.sidebar.selectbox(
    "How would you like to input features?",
    ("Single", "Batch"))

    st.sidebar.markdown("""
    [Example CSV input file](https://github.com/LangatGilbert/100daysofcode/blob/master/Employee%20turnover%20prediction/data/example_hr.csv)
    """)

    st.title("Employee Churn Prediction App")

    st.image(image_churn)

    st.write("""
    Photo by [Kate.sade](https://unsplash.com/photos/2zZp12ChxhU) on Unsplash.
    
    The model outputs 0 meaning the employee stays with the company and 1 means employee left.
    
    """
    
    )


    if add_selectbox == 'Single':

        satisfaction_level = st.sidebar.number_input('Satisfaction Level', min_value=0.1, max_value=1.0, value=0.5)
        last_evaluation = st.sidebar.number_input('Last Evaluation', min_value=1, max_value=100, value=25)
        number_project = st.sidebar.number_input('Projects', min_value=1, max_value=50, value=10)
        average_montly_hours = st.sidebar.number_input('Average Monthly Hours', min_value=50, max_value=400, value=200)
        time_spend_company = st.sidebar.number_input('Time Spent', min_value=1, max_value=30, value=10)
        Work_accident =st.sidebar.selectbox('Work Accident', [0, 1])
        promotion_last_5years = st.sidebar.selectbox('Promotion', [0, 1])
        dept = st.sidebar.selectbox('Department',['accounting','hr','IT','management','marketing','product_mng','RandD','sales','support','technical'])
        salary = st.sidebar.selectbox('Salary',  ['high','low','medium'])

        output=""

        input_dict = {'satisfaction_level' : satisfaction_level, 'last_evaluation' : last_evaluation, 'number_project' : number_project,
            'average_montly_hours' : average_montly_hours, 'time_spend_company' : time_spend_company, 'Work_accident' : Work_accident,'promotion_last_5years':promotion_last_5years,
            'dept':dept,'salary':salary}

        input_df = pd.DataFrame([input_dict])

        if st.sidebar.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = 'Label = ' + str(output)

        #st.success('The output is {}'.format(output))

        st.subheader("Model Prediction")
        st.write(output)
        st.write('---')

        

    if add_selectbox == 'Batch':

        file_upload = st.sidebar.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
    
            #checking model accuracy on the unseen dataset
            st.subheader("Model Predictions on the batch data")

            st.write(predictions.head())

            st.subheader("Model Accuracy")

            st.write(check_metric(predictions['left'], predictions['Label'], metric = 'Accuracy'))
            st.write('----')

            #Download the csv file.
            def filedownload(df):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href = "data:file/csv;base64,{b64}" download = "predictions.csv">Download the Predictions</a>'
                return href
                
            st.markdown(filedownload(predictions), unsafe_allow_html = True)
    
    
    st.sidebar.info('This app is created by Gilbert Langat to predict employees churn in organization XYZ. Data used is app is obtained from [Kaggle](https://www.kaggle.com/arvindbhatt/hrcsv)')
    st.sidebar.image(image)
    st.sidebar.success('https://www.pycaret.org')
if __name__ == '__main__':
    run()