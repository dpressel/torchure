--[[ 
   This implementation of a DataStore is file backed, one file per item,
   and these are serialized to disk in a directory (typically a temp dir).
   On request, these are loaded back to memory and handed to the consumer
]]--
local FileBackedStore, parent = torch.class('FileBackedStore', 'DataStore')

function FileBackedStore:__init(dir)
   parent.__init(self)
   self.nitems = 0
   self.dir = dir or paths.tmpname()
   if paths.dirp(self.dir) == false then
      print('Creating path ' .. self.dir)
      paths.mkdir(self.dir)
   end

end

function FileBackedStore:size()
   return self.nitems
end

function getFileNameForItem(dir, i)
   return dir .. '/item-' .. i .. '.ser'
end

function FileBackedStore:get(i)
   -- Return ith item
   return torch.load(getFileNameForItem(self.dir, i))
end

function FileBackedStore:put(item)
   -- add a new item, return new size
   self.nitems = self.nitems + 1
   torch.save(getFileNameForItem(self.dir, self.nitems), item)
end

