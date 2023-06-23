from flask import Flask, render_template, request, send_file, url_for
from model import CustomTextClassifier
from process_utils import predict, clear_text
from torch import load as torchload
from torch import device as torchdevice
import os

app = Flask(__name__)

classifier = CustomTextClassifier('cointegrated/rubert-tiny', 'cointegrated/rubert-tiny', n_classes=3)
classifier.model = torchload('/Users/iangustov/Downloads/bert_val_acc=0.85.pt', map_location=torchdevice('cpu'))

idx2labels = {
    0: 'Требования',
    1: 'Обязанности',
    2: 'Условия'
 }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    input_text = request.form['inputText']

    input_text = clear_text(input_text)
    classes = predict(classifier, input_text, idx2labels)

    duties = '.\n'.join(classes['Обязанности'])
    conditions = '.\n'.join(classes['Условия'])
    requirements = '.\n'.join(classes['Требования'])

    return render_template('result.html', duties=duties, conditions=conditions, requirements=requirements)

@app.route('/help')
def support():
    return render_template('help.html')

@app.route('/download', methods=['POST'])
def download():
    duties = request.form['duties']
    conditions = request.form['conditions']
    requirements = request.form['requirements']
    filename = request.form['filename'] + ".txt"
    result_text = f"Должностные обязанности: {duties}\nУсловия: {conditions}\nТребования к соискателю: {requirements}"
    with open(filename, 'w') as file:
        file.write(result_text)
    return send_file(filename, as_attachment=True)

import pandas as pd

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename.endswith('.csv'):
        file.save(file.filename)
        df = pd.read_csv(file.filename)
        if 'Должностные обязанности' in df.columns:
            input_text = ' '.join(df['Должностные обязанности'].tolist())
            input_text = clear_text(input_text)
            classes = predict(classifier, input_text, idx2labels)

            conditions = '.\n'.join(classes['Условия'])
            requirements = '.\n'.join(classes['Требования'])

            df['Условия'] = conditions
            df['Требования к соискателю'] = requirements

            output_filename = 'output.csv'
            df.to_csv(output_filename, index=False)
            return send_file(output_filename, as_attachment=True)
        else:
            return "Файл CSV не содержит столбец 'Должностные обязанности'."
    else:
        return "Неправильный формат файла. Пожалуйста, загрузите файл в формате CSV."


def result_from_text(text):
    duties = text[::3]
    conditions = text[1::3]
    requirements = text[2::3]
    return render_template('result.html', duties=duties, conditions=conditions, requirements=requirements)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

