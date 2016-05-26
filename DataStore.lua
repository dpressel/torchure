require 'paths'

--[[
   This is about the simplest data access encapsulation interface ever written.
   It does not try to be a data loader, that is up to the producer
   It also doesnt know anything about what it is holding, which makes it
   dead simple.

   In general, this API should work when you have a batch for training, and
   the batch needs to be handed to the consumer at the appropriate time.
   In those cases, you might have, for example, a table of tensors, one for
   the x value (feature vector), and one for the y value
   
--]]

local DataStore = torch.class('DataStore')

function DataStore:__init()
   -- Constructor
end

function DataStore:size()
end

function DataStore:get(i)
end

function DataStore:put(item)
end
