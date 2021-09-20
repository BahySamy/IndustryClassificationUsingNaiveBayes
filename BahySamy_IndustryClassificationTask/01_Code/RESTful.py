from flask import Flask, jsonify, request, render_template
from joblib import load
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

def ValuePredictor(to_predict):
    count_vector = load('count_vector.joblib')
    model = load('model.joblib')

    k = count_vector.transform(to_predict)
    result = model.predict(k)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_view = request.form.to_dict()
        to_predict_list = list(to_predict_view.values())
        
        result = ValuePredictor(to_predict_list)
        if result == 0:
            prediction = 'IT'
        elif result == 1:
            prediction = 'Marketing'
        elif result == 2:
            prediction = 'Education'
        else:
            prediction = 'Accountancy'
        return render_template("result.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
