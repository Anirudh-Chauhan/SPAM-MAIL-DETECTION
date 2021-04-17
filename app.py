from flask import Flask,render_template,request
import pickle

cv = pickle.load(open('cv.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':     
        msg = request.form['message']
        data = [msg]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
        return render_template('result.html',prediction=pred)
if __name__ == "__main__":
    app.run(debug=True)
