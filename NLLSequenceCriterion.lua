-- Only count unmasked items
local nn = require 'nn'
local NLLSequenceCriterion, parent = torch.class('NLLSequenceCriterion', 'nn.Criterion')

local PAD_MIN = 0

-- The target output may use a 0 value by default.  If you want to use
-- something else for pad, 1 is also fairly common, just set minpad=1
function NLLSequenceCriterion:__init(minpad)
   parent.__init(self)
   self.minpad = minpad or PAD_MIN
   self.gradInput = {}
end

function NLLSequenceCriterion:updateOutput(input, target)
   self.gradInput = {}
   assert(torch.type(target) == 'table', "expecting target table")
   assert(#target == #input, "target should have as many elements as input")

   self.output = 0
   valid = 0

   local nStep = #input
   local bSz = input[1]:size(1)
   local vSz = input[1]:size(2)
   for i=1,nStep do
      local in_i = input[i]

      local indices = target[i]
      local grad = typ0s(in_i)

      for j=1,bSz do
	 if indices[j] > self.minpad then
	    valid = valid + 1
	    self.output = self.output - in_i[{j, indices[j]}]
	    grad[{j, indices[j]}] = -1
	 end
      end
      table.insert(self.gradInput, grad)
   end

   self.output = self.output / valid
   for i=1,nStep do
      self.gradInput[i] = self.gradInput[i] / valid
   end

   return self.output
end

function NLLSequenceCriterion:updateGradInput(input, target)
   return self.gradInput
end
