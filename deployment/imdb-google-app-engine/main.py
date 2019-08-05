from flask import Flask, render_template, url_for, request, jsonify, Response
from waitress import serve
import aiohttp
import asyncio
import async_timeout
import sqlalchemy

import logging
import sys
import os
import datetime

from scipy.special import softmax
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from argparse import Namespace

from fastai import *
from fastai.text import *
from pathlib import Path

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)


from utils_inference import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, InputExample, do_inference, load_example)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}

path = os.path.dirname(os.path.abspath(__file__))
path=Path(path)
pytorch_model_path=path/"model/"
fastai_model_path=path/"model-fastai/"

args = Namespace(task_name='imdb', model_type='bert', \
    model_name_or_path=str(pytorch_model_path), max_seq_length=512, \
    do_lower_case = 'true', text="")

model = None
tokenizer = None

'''task_name = 'imdb'
model_type = 'bert'
model_name_or_path = 'https://storage.googleapis.com/deployment-247905.appspot.com/models/imdb_output_final_bert/'
max_seq_length = 512
classes = ['0', '1']'''

classes = ['neg', 'pos']
fastai_clas_path_url = "https://storage.googleapis.com/deployment-247905.appspot.com/models/imdb_ulmfit/data_clas.pkl"
fastai_model_path_url = "https://storage.googleapis.com/deployment-247905.appspot.com/models/imdb_ulmfit/final.pth"
urls = [fastai_clas_path_url, fastai_model_path_url]
destinations = [fastai_model_path/'data_clas.pkl', fastai_model_path/'models'/'final.pth']

logger = logging.getLogger(__name__)

db_user = os.environ.get("DB_USER")
db_pass = os.environ.get("DB_PASS")
db_name = os.environ.get("DB_NAME")
cloud_sql_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")

app = Flask(__name__)
if os.path.isfile(".debug"):    
    print("Running in debug mode")
    app.debug = True
else:
    print('NOT running in debug mode')

def load_pytorch_model():
    global tokenizer, model
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    
    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)   
    return model
        
load_pytorch_model()
logger.info("pytorch-transformers model loaded.")

async def download_file(url, dest, session):    
    if dest.exists(): 
        logger.info("File %s already exists.", dest)
        return 'File exists ' + str(dest)
    else:
        logger.info("Downloading file %s to %s", url, dest)
        async with async_timeout.timeout(900):
            async with session.get(url) as response:
                with open(dest, 'wb') as fd:
                    async for data in response.content.iter_chunked(1024):
                        fd.write(data)
        logger.info("Successfully downloaded %s.", dest)                
        return 'Successfully downloaded ' + str(dest)
       
async def setup_learner(urls, destinations):    
    async with aiohttp.ClientSession() as session:
        tasks = [download_file(url, destination, session) for url, destination in zip(urls, destinations)]
        return await asyncio.gather(*tasks)

def load_fastai_model():
    try:        
        data_clas = load_data(fastai_model_path, "data_clas.pkl", bs=48)
        learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
        learn.load('final')
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
results = loop.run_until_complete(setup_learner(urls, destinations))
'''print('\n'.join(results))'''

learn = load_fastai_model()
logger.info("Fastai model loaded.")

# [START cloud_sql_mysql_sqlalchemy_create]
# The SQLAlchemy engine will help manage interactions, including automatically
# managing a pool of connections to your database
db = sqlalchemy.create_engine(
    # Equivalent URL:
    # mysql+pymysql://<db_user>:<db_pass>@/<db_name>?unix_socket=/cloudsql/<cloud_sql_instance_name>
    sqlalchemy.engine.url.URL(
        drivername='mysql+pymysql',
        username=db_user,
        password=db_pass,
        database=db_name,
        query={
            'unix_socket': '/cloudsql/{}'.format(cloud_sql_connection_name)
        }
    ),
    # ... Specify additional properties here.
    # [START_EXCLUDE]

    # [START cloud_sql_mysql_sqlalchemy_limit]
    # Pool size is the maximum number of permanent connections to keep.
    pool_size=5,
    # Temporarily exceeds the set pool_size if no connections are available.
    max_overflow=2,
    # The total number of concurrent connections for your application will be
    # a total of pool_size and max_overflow.
    # [END cloud_sql_mysql_sqlalchemy_limit]

    # [START cloud_sql_mysql_sqlalchemy_backoff]
    # SQLAlchemy automatically uses delays between failed connection attempts,
    # but provides no arguments for configuration.
    # [END cloud_sql_mysql_sqlalchemy_backoff]

    # [START cloud_sql_mysql_sqlalchemy_timeout]
    # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
    # new connection from the pool. After the specified amount of time, an
    # exception will be thrown.
    pool_timeout=30,  # 30 seconds
    # [END cloud_sql_mysql_sqlalchemy_timeout]

    # [START cloud_sql_mysql_sqlalchemy_lifetime]
    # 'pool_recycle' is the maximum number of seconds a connection can persist.
    # Connections that live longer than the specified amount of time will be
    # reestablished
    pool_recycle=1800,  # 30 minutes
    # [END cloud_sql_mysql_sqlalchemy_lifetime]

    # [END_EXCLUDE]
)
# [END cloud_sql_mysql_sqlalchemy_create]

'''
@app.before_first_request
def create_tables():
    # Create tables (if they don't already exist)
    with db.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS votes "
            "( vote_id SERIAL NOT NULL, time_cast timestamp NOT NULL, "
            "review_text VARCHAR(2048) NOT NULL, model VARCHAR(16) NOT NULL, action VARCHAR(8) NOT NULL, "
            "client_ip VARCHAR(16), username VARCHAR(32), PRIMARY KEY (vote_id) );"
        )
'''

@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    args.text = request.form['review_text']
    args.text = args.text.replace('\n', ' ').replace('\r', '')
    logger.info("Input text = <%s>", args.text)    

    # Run the pytorch-transformers model
    pt_pred, pt_confidence = do_inference(args, model, tokenizer)
    pt_prediction = "Positive" if pt_pred==1 else "Negative"
    
    # Run the fastai model
    fa_pred = learn.predict(args.text)    
    print("fa_pred:", fa_pred)
    fa_prediction = "Positive" if str(fa_pred[0])=='pos' else "Negative"
    fa_confidence = int(fa_pred[2][1]*100) if fa_prediction=='Positive' else int(fa_pred[2][0]*100) 

    resp_dic = {'pt': {'prediction': pt_prediction, 'confidence': round(pt_confidence), 'review_text': args.text},
                'fa': {'prediction': fa_prediction, 'confidence': fa_confidence, 'review_text': args.text}}

    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'

    return resp

@app.route('/model-feedback', methods=['POST'])
def model_feedback():
    content = request.get_json()    
    model = content['model']
    review_text = content['review_text'].strip()

    if review_text == "":
        return Response(
            status=200,
            response="No review text found"
                     
        )

    action = content['action']
    time_cast = datetime.datetime.utcnow()
    client_ip = request.access_route[0]
    # client_ip = request.environ['REMOTE_ADDR']
    '''
    if request.headers.getlist("X-Forwarded-For"):
        # client_ip = request.headers.getlist("X-Forwarded-For")[0]
        # client_ip = request.headers.getlist("X-Forwarded-For").split(',')[0]
        client_ip = request.access_route[0]
    else:
        client_ip = request.remote_addr
    '''

    logger.info("[%s]: Action [%s] for model [%s] for review [%s] for [%s]", \
        time_cast.strftime('%B %d %Y - %H:%M:%S'), action, model, review_text, client_ip)

    # [START cloud_sql_mysql_sqlalchemy_connection]
    # Preparing a statement before hand can help protect against injections.
    stmt = sqlalchemy.text(
        "INSERT INTO votes (time_cast, review_text, model, action, client_ip)"
        " VALUES (:time_cast, :review_text, :model, :action, :client_ip)"
    )
    try:
        # Using a with statement ensures that the connection is always released
        # back into the pool at the end of statement (even if an error occurs)
        with db.connect() as conn:
            conn.execute(stmt, time_cast=time_cast, review_text=review_text, model=model, action=action, client_ip=client_ip)
    except Exception as e:
        # If something goes wrong, handle the error in this section. This might
        # involve retrying or adjusting parameters depending on the situation.
        # [START_EXCLUDE]
        logger.exception(e)
        return Response(
            status=500,
            response="Unable to register vote! Please check the "
                     "application logs for more details."
        )
        # [END_EXCLUDE]
    # [END cloud_sql_mysql_sqlalchemy_connection]

    return Response(
        status=200,
        response="Vote successfully cast for '{}' at time {}!".format(
            client_ip, time_cast)
    )
    '''
    resp_dic = {'status': 'success'}
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp'''

if __name__ == '__main__':
    if app.debug:
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        serve(app, host='0.0.0.0', port=8080)
	

    


