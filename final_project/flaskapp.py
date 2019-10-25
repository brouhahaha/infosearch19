from flask import Flask
from flask import render_template
from flask import request
import urllib.request

import search

import logging
import logging.handlers

handler = logging.handlers.RotatingFileHandler(
        'log.txt',
        maxBytes=1024 * 1024)

logging.getLogger('werkzeug').setLevel(logging.DEBUG)
logging.getLogger('werkzeug').addHandler(handler)

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(handler)

@app.route('/')
def index():
    app.logger.info('someone accessed search_page')
    return render_template(
        'search_page.html',
        data=[{'name':'tf-idf'}, {'name':'BM25'}, {'name':'fasttext'}, {'name':'ELMO'}])

@app.route("/results" , methods=['GET', 'POST'])
def results():
    query = request.args['search']
    model = request.args['model']
    if model == 'BM25':
        app.logger.info('%s model is being used', model)
        matrix = search.get_bm25matrix(r'C:\Users\Maria\Desktop\infosearch_f\bm25.csv')
        result = search.get_result(query, matrix)
    elif model == 'tf-idf':
        app.logger.info('%s model is being used', model)
        matrix = search.get_pickle_matrix(r'C:\Users\Maria\Desktop\infosearch_f\tf.pickle')
        result = search.get_result(query, matrix)
    elif model == 'fasttext':
        app.logger.info('%s model is being used', model)
        matrix = search.get_pickle_matrix(r'C:\Users\Maria\Desktop\infosearch_f\ftext.pickle')
        result = search.get_fasttext_res(matrix, query, r'C:\Users\Maria\Desktop\infosearch_f\model.model')
    app.logger.info('search results for '+query+' are given')
    result = [k[0] for k in result]
    return render_template(
        'result_page.html', query = query, model = model, resultp = result)

if __name__=='__main__':
    
    app.run(debug=True)
