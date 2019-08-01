from flask import Flask, render_template, request

app = Flask(__name__)
"""
Your functions here

"""

@app.route('/') # These are your routes or pages
def home():
	return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
