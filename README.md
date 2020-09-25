# Bibliographic Reference Parser for German 
## Model
<img src="misc/model.png" width="200"/>
The model consists of two bidirectional GRUs and two dense layers. The first GRU receives the output of the last layer of a multilingual BERT model as input.
It is not sufficient to use a German BERT model, because it cannot be adapted to the mostly English data from the <a href="https://github.com/kermitt2/grobid">GROBID</a> project. The Multilingual Model on the other hand can be trained on both English and German data and achieves better results in combination.
## Training
<img src="misc/train.png" width="300"/>

## Usage
