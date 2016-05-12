local nn = require 'nn'
local Word2VecLookupTable, parent = torch.class('Word2VecLookupTable', 'nn.LookupTable')


function Word2VecLookupTable:__init(filename, knownvocab, unifweight, normalize)
    parent.__init(self, 0, 0)
    local uw = unifweight or 0.0
    file = torch.DiskFile(filename, 'r')
    file:binary()
    -- |V| x d
    local vxd = readupto(file, '\n'):split(' ')
    self.vsz = 0
    local vsz = tonumber(vxd[1])
    if knownvocab then
       for name,_ in pairs(knownvocab) do
	  self.vsz = self.vsz + 1
       end
    else
       self.vsz = vsz
    end

    self.dsz = tonumber(vxd[2])
    self.weight = torch.FloatTensor(self.vsz+1, self.dsz):uniform(-uw, uw)
    self.gradWeight = torch.FloatTensor(self.vsz+1, self.dsz):zero()
    
    -- loop |V| times read in vocab to table
    rv = {}
    self.vocab = {}
    
    -- Make sure padding is zero
    self.weight[1]:zero()
    self.vocab["<PADDING>"] = 1

    local k = 2
    for i=1,vsz do

       -- read in word       
       local word = readupto(file, ' ')

       -- read its vec
       vec = torch.FloatTensor(file:readFloat(self.dsz))

       if knownvocab == nil or knownvocab[word] then
	  self.vocab[word] = k
	  -- normalize the vector!
	  self.weight[{{k},{}}] = normalize and vec:div(vec:norm()) or vec
	  k = k + 1
	  knownvocab[word] = nil
       end
    end

    -- for any remaining words in the knownvocab
    if knownvocab then
       for name,_ in pairs(knownvocab) do
	   self.vocab[name] = k
	   k = k + 1
       end
    end
    self.vsz = self.vsz + 1
    file:close()

end

function Word2VecLookupTable:lookup(word)
    return self.weight[self.vocab[word]]
end

function Word2VecLookupTable:sim(word1, word2)
    v1 = self.weight[self.vocab[word1]]
    v2 = self.weight[self.vocab[word2]]
    return v1:dot(v2)
end
