import pickle
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Loading the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_model.sav', 'rb'))
cancer_model = pickle.load(open('cancer_model.sav', 'rb'))

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_disease', methods=['POST'])
def select_disease():
    selected_disease = request.form['disease']

    if selected_disease == 'Diabetes':
        return render_template('diabetes_form.html')
    elif selected_disease == 'Heart Disease':
        return render_template('heart_disease_form.html')
    elif selected_disease == 'Cancer':
        return render_template('cancer_form.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    # Retrieve form data for diabetes prediction
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])

    # Make a prediction
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    if diab_prediction[0] == 0:
        diab_diagnosis = 'The person is not diabetic'
    else:
        diab_diagnosis = 'The person is diabetic'

    return render_template('index.html', result_type='Diabetes', result=diab_diagnosis)

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    # Retrieve form data for heart disease prediction
    # Parse form fields and convert to appropriate data types
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    # Make a prediction
    heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if heart_prediction[0] == 1:
        heart_diagnosis = 'The person is having heart disease'
    else:
        heart_diagnosis = 'The person does not have any heart disease'

    return render_template('index.html', result_type='Heart Disease', result=heart_diagnosis)

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    # Retrieve form data for cancer prediction
    # Parse form fields and convert to appropriate data types
    clump_thickness = float(request.form['clump_thickness'])
    unif_cell_size = float(request.form['unif_cell_size'])
    unif_cell_shape = float(request.form['unif_cell_shape'])
    marg_adhesion = float(request.form['marg_adhesion'])
    single_epith_cell_size = float(request.form['single_epith_cell_size'])
    bare_nuclei = float(request.form['bare_nuclei'])
    bland_chromation = float(request.form['bland_chromation'])
    normal_nucleoli = float(request.form['normal_nucleoli'])
    mitoses = float(request.form['mitoses'])

    # Make a prediction
    cancer_prediction = cancer_model.predict([[clump_thickness, unif_cell_size, unif_cell_shape, marg_adhesion, single_epith_cell_size, bare_nuclei, bland_chromation, normal_nucleoli, mitoses]])

    if cancer_prediction[0] == 4:
        cancer_diagnosis = 'The person has Breast Cancer'
    else:
        cancer_diagnosis = 'The person does not have Breast Cancer'

    return render_template('index.html', result_type='Cancer', result=cancer_diagnosis)

if __name__ == '__main__':
    app.run(debug=True)

