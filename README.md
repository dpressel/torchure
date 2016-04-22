# emb
Tiny module to load and use existing Word2Vec embeddings in Torch

About
-----

Load and use word embeddings.  Currently just reads word2vec binaries.  There are two ways to use these.  You can load up just a representation of the embeddings into "Word2VecModel":

```

th> require 'emb';
                                                                      [0.0000s]	
th> wv = Word2VecModel('/data/xdata/oct-s140clean-uber.cbow-bin')
                                                                      [14.4513s]
th> wv.vsz, wv.dsz  
949887	150		
                                                                      [0.0001s]	
th> hellov = wv:word2vec('hello')

```

If a word isnt present, by default, it gives back a zero-vector.  If you pass true to the second parameter, it will return nil instead

```
      
th> wv:word2vec('kjlasgjklwljk', true)
                                                                      [0.0000s]	
th> wv:word2vec('kjlasgjklwljk', false)
 0
 0
 0
 0
 0
 0

```

The approach above might be useful if you are not planning on fine-tuning your embeddings and want to just preprocess your data.  For instance, in the case of classification, you can simply form a tensor (or batch of tensors).  This example loads a temporal vector from a TSV, where the first tab is the label, followed by a sentence.  It returns a table with-sub tables for x (the feature vector) and y (the label).  Each feature vector has a row for each word, and a column for each embedding dimension.

```
function loadTemporal(file, w2v, filtsz, mxlen)
    local ts = {}
    local yt = {}
    local xt = {}

    local vsz = w2v.vsz
    local dsz = w2v.dsz
    local fsz = filtsz or 0
    local mxw = mxlen or 128
    local halffiltsz = math.floor(fsz / 2)

    local tsfile = io.open(file, 'r')
    local linenum = 1

    for line in tsfile:lines() do  
	local labelText = line:split('\t')
	if #labelText < 2 then
	   print('Skipping invalid line ' .. line .. " " .. linenum)
	   goto continue 
	end

	local num = tonumber(labelText[1])
	local y = torch.FloatTensor({num})
	local toks = labelText[2]:split(' ')

	local mx = math.min(#toks, mxw)
        local siglen = mx + (2*halffiltsz)
 	local x = torch.zeros(siglen, dsz)
	for i=1,mx do
	    local w = toks[i]
	    local z = w2v:word2vec(w)
	    x[{i + halffiltsz}] = z
	end
	table.insert(yt, y)
	table.insert(xt, x)
        
	linenum = linenum + 1
	::continue::
    end
    ts.y = yt
    ts.x = xt
    return ts
end
```

If you want to just use a model directly as pretraining for a nn.LookupTable, you can use the Word2VecLookupTable instead.  The usage is the same (almost), except this can be used both as a way to find the word vectors for indices, and it can be put directly into a neural network.

```
th> wv = Word2VecLookupTable('/data/xdata/oct-s140clean-uber.cbow-bin')
                                                                      [12.4409s]	
th> wv.vsz, wv.dsz
949887	150	
                                                                      [0.0001s]	
th> hellov = wv:lookup('hello')
                                                                      [0.0001s]	
th> wv:lookup('kjlasgjklwljk')
                                                                      [0.0000s]	
```
Now we can use the table directly as a LookupTable, for example, in a convolution network  

```
function createNN(w2v, cmotsz, filtsz, nc, gpu)
    local seq = nn.Sequential()
    seq:add(w2v)
    seq:add(nn.TemporalConvolution(w2v.dsz, cmotsz, filtsz))
    seq:add(nn.Max())
    seq:add(nn.Linear(cmotsz, nc))
    seq:add(nn.LogSoftMax())

    return gpu and seq:cuda() or seq
end

local w2v = Word2VecLookupTable(embeddings)
print('Loaded word embeddings')
local cnn = createCNN(w2v, 200, 5, labelsz, gpu)
```

Installing
----------

*From git*

- clone the repository

- execute:
```
luarocks make emb-scm-1.rockspec
```

*From web*

- execute:
```
luarocks install https://raw.githubusercontent.com/dpressel/emb/master/emb-scm-1.rockspec
```

