from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import logging
import sys
import os
from pathlib import Path

import numpy as np
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
    model_name_or_path='/home/nikhil_subscribed/nlp_projects/pytorch-transformers-extensions/examples/imdb_output_final_bert/', max_seq_length=512, \
    do_lower_case = 'true', text="")

model = None
tokenizer = None

'''task_name = 'imdb'
model_type = 'bert'
model_name_or_path = 'https://storage.googleapis.com/deployment-247905.appspot.com/models/imdb_output_final_bert/'
max_seq_length = 512
classes = ['0', '1']'''

path = Path(__file__).parent

logger = logging.getLogger(__name__)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

'''async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)'''

async def load_model():
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
        

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(load_model())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    print("inside analyze......")
    logger.info("Data: %s", data)
    args.text = data['review_text']
    pred, confidence = do_inference(args, model, tokenizer)

    return HTMLResponse(
        """
        <html>
           <body>
             <p>Prediction: <b>%d</b></p>
             <p>Confidence: %d</p>
           </body>        
        </html>
    """ %(pred, confidence))

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8000)

