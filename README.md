# babyGPT
re-implementation of GPT.

## Install
```
pip install -r requirements.txt
```

## Data processing
```
python data/wikitext/prepare.py
```

## Training
```
python train.py
```
After one-epoch, you will get ~2.8 cross-entropy loss on training data.

## Generate
```
python generate.py
```
An example will be:
```
Enter prompt: This ammunition , and that which I brought with me , was rapidly prepared for use at the Laboratory established at the Little
This ammunition , and that which I brought with me , was rapidly prepared for use at the Laboratory established at the Little Rock Department of Anthropology and Political Science . 
<|endoftext|> In the early 20th century , the earliest settlers in the English history of the Second World War , the Irish @-@ Scots settlers were established in the area of the Anglo @-@ Irish ascendancy , which the Crown by the Cyrillic , according to the historian of the legendary spot where the town was created by the origin of the Roman Empire . The historian Cretusa was a position known as the " labyrinth " , " is that is " labyrinth has
```

## References
[minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT)
