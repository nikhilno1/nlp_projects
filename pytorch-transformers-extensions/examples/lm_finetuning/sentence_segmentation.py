from argparse import ArgumentParser
from pathlib import Path
import os
import re
import spacy

def remove_html_tags(text):
    """Remove html tags from a string"""    
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def clean_text(text):
    """Apply different cleanup mechanisms"""
    text = remove_html_tags(text)
    return text

def create_corpus_file(input_dir, output_file):
    nlp = spacy.load('en_core_web_sm')
    o_fh = open(output_file, 'w')

    """Remove HTML tokens"""
    remove_html = re.compile('<.*?>')
    remove_quotes = re.compile('"+')
    remove_asterisks = re.compile('\*+')

    corpus = ""    

    for input_file in os.listdir(input_dir):
        if input_file.endswith(".txt"):
            with open(os.path.join(input_dir, input_file), "r") as i_fh:
                text = i_fh.readlines()[0]
                text = re.sub(remove_html, '', text)
                text = re.sub(remove_quotes, '', text)
                text = re.sub(remove_asterisks, '', text)
                text_sentences = nlp(text)
                                
                for sentence in text_sentences.sents:
                    """print(sentence.text)"""
                    corpus += sentence.text
                    corpus += '\n'
                    """o_fh.write(sentence.text + '\n')"""
        """o_fh.write('\n')"""
        corpus += '\n'
    o_fh.write(corpus)            

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    
    args = parser.parse_args()

    create_corpus_file(args.input_dir, args.output_file)

if __name__ == '__main__':
    main()
