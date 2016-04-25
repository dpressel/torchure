local nn = require 'nn'
local Word2VecLookupTable, parent = torch.class('Word2VecLookupTable', 'nn.LookupTable')


function Word2VecLookupTable:__init(filename)
    parent.__init(self, 0, 0)
    file = torch.DiskFile(filename, 'r')
    file:binary()
    -- |V| x d
    local vxd = readupto(file, '\n'):split(' ')
    self.vsz = tonumber(vxd[1])
    self.dsz = tonumber(vxd[2])
    self.weight = torch.FloatTensor(self.vsz+1, self.dsz)
    self.gradWeight = torch.FloatTensor(self.vsz+1, self.dsz):zero()
    
    -- loop |V| times read in vocab to table
    rv = {}
    self.vocab = {}

    for i=1,self.vsz do

       -- read in word       
       local word = readupto(file, ' ')

       -- read its vec
       vec = torch.FloatTensor(file:readFloat(self.dsz))

       self.vocab[word] = i

       -- normalize the vector!
       self.weight[{{i},{}}] = vec:div(vec:norm())
    end
    
    self.vsz = self.vsz + 1
    self.vocab["<PADDING>"] = self.vsz
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
