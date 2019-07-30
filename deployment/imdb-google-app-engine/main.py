from waitress import serve
from flask import Flask, render_template, url_for, request
import logging
import sys
import os

from scipy.special import softmax
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

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

from argparse import Namespace
args = Namespace(task_name='imdb', model_type='bert', \
    model_name_or_path='app/model/', max_seq_length=512, \
    do_lower_case = 'true', text="")

model = None
tokenizer = None

'''task_name = 'imdb'
model_type = 'bert'
model_name_or_path = 'https://storage.googleapis.com/deployment-247905.appspot.com/models/imdb_output_final_bert/'
max_seq_length = 512
classes = ['0', '1']'''


logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_model():
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
        
load_model()

logger.info("Model loaded.")

@app.route('/')
def index():    
    return render_template('index.html', prediction="", confidence=50, review_text="")

@app.route('/analyze', methods=['POST'])
def analyze():
    args.text = request.form['review_text']
    args.text = args.text.replace('\n', ' ').replace('\r', '')
    logger.info("Input text = <%s>", args.text)    
    pred, confidence = do_inference(args, model, tokenizer)
    prediction = "Positive" if pred==1 else "Negative"
    
    return render_template('index.html', prediction=prediction, confidence=round(confidence), review_text=args.text)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
	

    


