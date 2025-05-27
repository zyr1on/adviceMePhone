from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

def load_phone_data():
    try:
        df = pd.read_csv('phone_dataset.csv')
        return df
    except:
        return None

def filter_phones(query):
    df = load_phone_data()
    if df is None:
        return []
    
    query = query.lower().strip()
    
    if 'android' in query:
        filtered = df[df['brand'].str.lower() == 'android']
    elif 'iphone' in query:
        filtered = df[df['brand'].str.lower() == 'iphone']
    else:
        return []
    
    return filtered.to_dict('records')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_input():
    user_input = request.json.get('input', '')
    
    if not user_input.strip():
        return jsonify({'response': 'Lütfen bir şey yazınız.', 'phones': []})
    
    phones = filter_phones(user_input)
    
    if phones:
        return jsonify({
            'response': f'{len(phones)} telefon bulundu:',
            'phones': phones,
            'query': user_input
        })
    else:
        return jsonify({
            'response': 'Telefon bulunamadı. "Android" veya "iPhone" yazın.',
            'phones': [],
            'query': user_input
        })

if __name__ == '__main__':
    app.run(debug=True)
