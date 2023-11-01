from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    if request.method == 'GET':
        return jsonify({'message': 'This is a GET request.'})
    elif request.method == 'POST':
        try:
            data = request.json
            return jsonify({'message': 'This is a POST request.', 'data': data})
        except Exception as e:
            return jsonify({'error': 'Invalid JSON data.'}), 400

if __name__ == '__main__':
    app.run()
