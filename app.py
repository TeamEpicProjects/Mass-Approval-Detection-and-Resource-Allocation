import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

df = pd.read_csv("datasets/combined_data.csv")
Xt1 = pd.read_csv("datasets/app_test.csv")
df.dropna(inplace = True)
df.drop(['approval_status'], axis = 1, inplace = True)

df.drop(['request_id','approver', 'approved_on','requested_on','UID','Unnamed: 0'], axis = 1, inplace = True)

i=0
encoders=[]
name='encoder'
for col in df.columns:
    temp=name+str(i)
    globals()[temp] = ce.BinaryEncoder()
    globals()[temp] = globals()[temp].fit(df[col])
    encoders.append(globals()[temp])
    i=i+1

df2 = pd.read_csv("F:/Packt/Project - Chirasmita/clone/Mass-Approval-Detection-and-Resource-Allocation/datasets/user_data.csv")

df2.drop(['Unnamed: 0'],axis=1,inplace=True)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global output
    int_features = [x for x in request.form.values()]

    id=int_features[0]
    f_features=int_features[1:]
    temp=list(df2[df2['uid']==id].values[0])
    id=id.replace("requestee","account")
    f_features=f_features[:3]+[id]+f_features[3:]
    f_features=f_features+temp[1:]
    f_features = np.array(f_features).reshape(1, 11)
    f_features = pd.DataFrame(f_features, columns=df.columns)

    global features
    features = pd.DataFrame()
    for i in range(11):
        features = pd.concat([features,encoders[i].transform(f_features.iloc[:,i])], axis = 1)


    prediction = model.predict(features)

    output = prediction[0]
    return render_template('index.html', saved = 'saved:', features = int_features)


@app.route('/recommend',methods=['POST'])
def recommend():

    Xt1.iloc[:, 11:] = features.iloc[:,11:]
    y_proba1 = model.predict_proba(Xt1)
    y_prob1 = pd.DataFrame(y_proba1, index=df.application.unique(), columns=['A', 'R'])
    y_prob1 = y_prob1.sort_values('A', ascending=False)
    recommend = y_prob1.loc[y_prob1.A > 0.96].index.values
    if len(recommend) == 0:
        recommend = np.append(recommend, 'No applications recommended')

    if "Recommend" in request.form:
        return render_template('index.html',txt= 'Recommended apps:\n ', recommend = recommend)

    elif "Approval" in request.form:
        if output == 0:
            return render_template('index.html', prediction='Verdict: Accepted')

        else:
            return render_template('index.html', prediction='Verdict: Rejected')

if __name__ == "__main__":
    app.run(debug=True)

