-- http://lua-users.org/wiki/StringTrim
function trim(s)
  return s:match'^%s*(.*%S)' or ''
end

-- NaNs?
function hasNaN(t)
   mx = torch.max(t)
   return mx ~= mx
end

function readupto(file, ch)
    local word = ''
    local v = string.byte(ch)
    local ct = {}    
    while true do
        local ch = file:readChar()
	if ch == v or ch == 0 or ch == nil then break end
        table.insert(ct, ch)
    end
    local chars = torch.CharStorage(ct)

    return trim(chars:string())
end

-- Take a map[key] = index table, and make a map[index] = key
function revlut(f2i)
   local i2f = {}
   for k,v in pairs(f2i) do
      i2f[v] = k
   end
   return i2f
end

-- Reverse an array table
function revtab(tab)
    local size = #tab
    local newTable = {}

    for i,v in ipairs ( tab ) do
        newTable[size-i] = v
    end
    return newTable
end

-- WIP
function saveModel(model, file, gpu)
   if gpu then model:float() end
   torch.save(file, model)
   if gpu then model:cuda() end
end

function loadModel(file, gpu)
   local model = torch.load(file)
   return gpu and model:cuda() or model
end

function lookupSent(rlut, lu)
   local words = {}
   for i=1,lu:size(1) do
      local word = rlut[lu[i]]
      if word ~= '<PADDING>' then
	 table.insert(words, word)
      end
   end
   return table.concat(words, " ")
end

function numLines(file)
   local tsfile = io.open(file, 'r')
   local count = 0 
   for line in tsfile:lines() do
      count = count + 1
   end
   return count
end
