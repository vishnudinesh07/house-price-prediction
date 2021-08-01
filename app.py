from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle 

app = Flask(__name__)

filename = 'model.pkl'

# load model first 
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home_page():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

	data = [request.form['f1'],
			request.form['f2'],
			request.form['f3'],
			request.form['f4'],
			request.form['f5'],
			request.form['f6'],
			request.form['f7'],
			request.form['f8'],
			request.form['f9'],
			request.form['f10'],
			request.form['f11'],
			request.form['f12'],
			request.form['f13'],
			request.form['f14']
			]

	data = np.array([np.asarray(data, dtype=float)])
	print(data)
	if np.sum(data) ==  0:

		predictions =0
		
	else:

		predictions = loaded_model.predict(data)

	return render_template('index.html' , pred = predictions)


if __name__ == "__main__":

	app.run(host='127.0.0.1', port=8000 , debug =True)
