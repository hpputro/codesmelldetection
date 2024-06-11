import numpy as np
import pandas as pd
from XMLGrammar.LexGrammarBuilder import LexGrammarBuilder
from XMLGrammar.LexParser import LexParser

builder = LexGrammarBuilder("java")
lexGram = builder.read()
lexer = LexParser(lexGram)

df = pd.read_csv('magicvalue.csv')
filets = []
filesc = []
i = 0
for line in df['source']:
    lexer.resetTokens()
    lexer.parse(line)
    ts = lexer.getInStream()
    filets.append(ts)
    sc = lexer.getInSource()
    filesc.append(sc)
    i = i + 1
print(i)
label = {"label": {"clean": 0, "smell": 1}}
df.replace(label, inplace=True)

tokenstream = np.array(filets)
dfts = pd.DataFrame({'label': df['label'], 'source': tokenstream}, columns=['label', 'source'])
dfts.to_csv("ts.csv")
print(tokenstream.shape)

sourcecode = np.array(filesc)
dfsc = pd.DataFrame({'label': df['label'], 'source': sourcecode}, columns=['label', 'source'])
dfsc.to_csv("sc.csv")
print(sourcecode.shape)
