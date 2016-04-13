Similar to [https://github.com/dvictor/lstm-poetry](https://github.com/dvictor/lstm-poetry) but this time, instead of characters,
we represent entire words as (embedding) vectors.

### Uses
[TensorFlow](https://www.tensorflow.org/) version 0.7.1

### Run

 - `train.py` train your NN
 - `generate.py` generate text 
 
Change WORK_DIR in each file to specify your work directory. 


### Here's an example of output.

Input text:

*green people floating  
the morning has* 

Output:

green people floating  
the morning has just begun  

that's what the world has left to do   
if you don't care how it's gotta be  

i'd like to see you laughing at me  

i can hear you say,  

i can see you in the eyes of a smile   
i'll be standing by your side  

i can't stop the tears  
with it black <unk>  
i can see the white lines   

can i even get your love?  

---

Occasional `<unk>` in the output are caused by the limit imposed to the vocabulary size.  
The vocabulary is composed of `vocab_size` most used words in the training set.    
The words in the input that are not part of the vocabulary are encoded as `<unk>` so the NN
will also generate some when it "feels like".
