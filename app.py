import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/SaiRamMaganti/Work/Projects/Mass Approval Detection/combined_data.csv")
df.dropna(inplace = True)
df.drop(['approval_status'], axis = 1, inplace = True)

df.drop(['request_id','approver', 'approved_on','requested_on','UID','Unnamed: 0'], axis = 1, inplace = True)

i=0
encoders=[]
name='encoder'
for col in df.columns:
    temp=name+str(i)
    globals()[temp] = LabelEncoder()
    globals()[temp] = globals()[temp].fit(df[col])
    encoders.append(globals()[temp])
    i=i+1

df2 = pd.read_csv("C:/Users/SaiRamMaganti/Work/Projects/Mass Approval Detection/User_Data.csv")
df2.drop(['Unnamed: 0'],axis=1,inplace=True)

app = Flask(__name__)
model = pickle.load(open('final_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    id=int_features[0]
    f_features=int_features[1:]
    temp=list(df2[df2['uid']==id].values[0])
    id=id.replace("requestee","account")
    f_features=f_features[:3]+[id]+f_features[3:]
    f_features=f_features+temp[1:]

    features=[]
    for i in range(11):
        features.append(encoders[i].transform([f_features[i]])[0])


    prediction = model.predict(np.array([features]))

    output = prediction[0]
    if output==1:
        return render_template('index.html', prediction='Predicted Approval Status: Accepted')
    else:
        return render_template('index.html', prediction='Predicted Approval Status: Rejected')

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
