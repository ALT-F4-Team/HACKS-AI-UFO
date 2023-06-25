from flask import Flask, render_template, request, send_file, url_for, jsonify
from model import CustomTextClassifier
from process_utils import predict, clear_text
from torch import load as torchload
from torch import device as torchdevice
import pandas as pd
from random import choice
import os
import io

app = Flask(__name__)

randoms_vacs = list(pd.read_csv('final.csv')['responsobilities'])
classifier = CustomTextClassifier('cointegrated/rubert-tiny', 'cointegrated/rubert-tiny', n_classes=3)
classifier.model = torchload('/home/UFOHACK/models/ruBERT_by_ALTF4.pt', map_location=torchdevice('cpu'))

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

    duties = '\n'.join(classes['Обязанности'])
    conditions = '\n'.join(classes['Условия'])
    requirements = '\n'.join(classes['Требования'])

    return render_template('result.html', duties=duties, conditions=conditions, requirements=requirements)

@app.route('/get_random')
def return_vac():
    response = jsonify({'text': choice(randoms_vacs)})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/help')
def support():
    return render_template('help.html')

@app.route('/download', methods=['POST'])
def download():
    duties = request.form['duties']
    conditions = request.form['conditions']
    requirements = request.form['requirements']
    filename = 'output.xlsx'

    data = pd.DataFrame(data={'Обязанности': [duties], 'Требования': [requirements], 'Условия': [conditions]})
    data.to_excel(filename, index=False)

    return send_file(filename, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(io.BytesIO(file.read()))
    else:
        return "Неправильный формат файла. Пожалуйста, загрузите файл в формате CSV или Excel."

    if 'Должностные обязанности' in df.columns:
        data = []
        for input_text in df['Должностные обязанности']:
            input_text = clear_text(input_text)
            classes = predict(classifier, input_text, idx2labels)

            classes['Обязанности'] = '. '.join(classes['Обязанности'])
            classes['Условия'] = '. '.join(classes['Условия'])
            classes['Требования'] = '. '.join(classes['Требования'])

            data.append(classes)

        df = pd.DataFrame.from_records(data)

        output_filename = 'output'
        if file_extension == '.csv':
            output_filename += '.csv'
            df.to_csv(output_filename, index=False, encoding='windows-1251')
        else:
            output_filename += '.xlsx'
            df.to_excel(output_filename, index=False)

        return send_file(output_filename, as_attachment=True)
    else:
        return "Файл не содержит столбец 'Должностные обязанности'."


if __name__ == '__main__':
    app.run(host='0.0.0.0')
