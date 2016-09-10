local nn = require 'nn'
local VocabLookupTable, parent = torch.class('VocabLookupTable', 'nn.LookupTable')


function VocabLookupTable:__init(knownvocab, dsz, unifweight)
    parent.__init(self, 0, 0)
    local uw = unifweight or 0.0

    self.vocab = {}
    self.vsz = 1
    self.vocab["<PADDING>"] = 1

    for name,_ in pairs(knownvocab) do
       self.vsz = self.vsz + 1
       self.vocab[name] = self.vsz
    end

    self.dsz = dsz
    self.weight = torch.FloatTensor(self.vsz, self.dsz):uniform(-uw, uw)
    self.gradWeight = torch.FloatTensor(self.vsz, self.dsz):zero()
end

function VocabLookupTable:lookup(word)
    return self.weight[self.vocab[word]]
end

function VocabLookupTable:sim(word1, word2)
    v1 = self.weight[self.vocab[word1]]
    v2 = self.weight[self.vocab[word2]]
    return v1:dot(v2)
end
