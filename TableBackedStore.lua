local TableBackedStore, parent = torch.class('TableBackedStore', 'DataStore')

--[[
   This interface fulfills a memory backed implementation of a store
   It retrieves the item from an internal table
]]--
function TableBackedStore:__init()
   -- Constructor
   parent.__init(self)
   self.items = {}
end

function TableBackedStore:size()
   return #(self.items)
end

function TableBackedStore:get(i)
   -- Return ith item
   return self.items[i]
end

function TableBackedStore:put(item)
   table.insert(self.items, item)
end
