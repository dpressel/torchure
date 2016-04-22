local Word2VecModel = torch.class('Word2VecModel')

function Word2VecModel:__init(filename)
    file = torch.DiskFile(filename, 'r')
    file:binary()
    -- |V| x d
    local vxd = readupto(file, '\n'):split(' ')
    self.vsz = tonumber(vxd[1])
    self.dsz = tonumber(vxd[2])
    self.w2v = torch.FloatTensor(self.vsz, self.dsz)
    -- loop |V| times read in vocab to table
    self.vocab = {}
    -- zero makes most sense for now
    self.nullv = torch.zeros(self.dsz):float()

    for i=1,self.vsz do

       -- read in word       
       local word = readupto(file, ' ')

       -- read its vec
       vec = torch.FloatTensor(file:readFloat(self.dsz))
       self.vocab[word] = i

       self.w2v[{i}] = vec:div(vec:norm())
    end

    file:close()
end

function Word2VecModel:lookup(word, nullifabsent)
    local key = self.vocab[word]
    if key ~= nil then 
        return self.w2v[key]
    end
    if nullifabsent then return nil end
    return self.nullv
end

function Word2VecModel:sim(word1, word2)
    v1 = self.w2v[self.vocab[word1]]
    v2 = self.w2v[self.vocab[word2]]
    return v1:dot(v2)
end
