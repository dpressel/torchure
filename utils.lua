require 'lfs'

-- http://lua-users.org/wiki/StringTrim
function trim(s)
  return s:match'^%s*(.*%S)' or ''
end

function startswith(str, sub)
   local len = math.min(sub:len(), str:len())
   return str:sub(1, len) == sub
end

function endswith(str, sub)
   local strlen = str:len()
   local sublen = sub:len()
   local start = math.max(strlen - sublen + 1, 1)
   return str:sub(start) == sub
end

-- NaNs?
function hasNaN(t)
   local mx = torch.max(t)
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

-- Get a subtable from indices given
-- if from is not given, 1 is used
-- if to is not given, end is used
-- negative indexing allowed for to
function subtab(orig, from, to)
   local start = from or 1
   local T = #orig
   local upto = to or 0
   upto = upto > 1 and upto < T and upto or (T+upto)

   local tab = {}
   for i=start,upto do
      table.insert(tab, orig[i])
   end
   return tab
end

-- Prepend to table
function pretab(tab, item)
   local copy = {}
   table.insert(copy, item)
   for _,v in pairs(tab) do
      table.insert(copy, v)
   end
   return copy
end

function jointab(tab, delim)
   
   local joinval = delim and delim or ' '
   if #tab == 0 then 
      return ''
   end
   local s = tab[1]
   for i=2,#tab do 
      s = s .. joinval .. tab[i]
   end
  return s
end

function firstin(tab, x)
   for i,v in pairs(tab) do
      if v == x then
	 return i
      end
   end
   return 0
end

function dim1(obj)
   local sz = type(obj) == 'table' and #obj or obj:size()[1]
   return sz
end

function typ0s(t1)
   return torch.zeros(t1:size()):type(t1:type())
end

-- Make the Tensor into a table in the first dimension
-- if form is not given, 1 is used
-- if to is not given, end is used
-- negative indexing allowed for to
function tab1st(orig, from, to)

   local start = from or 1
   local T = orig:size()[1]
   local upto = to or 0
   upto = upto > 1 and upto < T and upto or (T+upto)

   local tab = {}
   for i=start,upto do
      table.insert(tab, orig[i])
   end
   return tab
end

function mkdirs(tgt_dir)
   if tgt_dir ~= '' then
      local atts = lfs.attributes(tgt_dir)
      if atts == nil or atts.mode ~= 'directory' then
	 print('No such directory, creating ' .. tgt_dir)
	 lfs.mkdir(tgt_dir)
      else
	 print('Detected directory ' .. tgt_dir)
      end
   end

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

local I2S_DEF_STOP_WORDS = {'<PAD>', '<PADDING>', '<GO>', '<EOS>'}
function indices2sent(index2word, indices, rev, stopwords)
   local words = {}
   local filt = stopwords or I2S_DEF_STOP_WORDS
   local sz = dim1(indices)
   for i=1,sz do
      local word = index2word[indices[i]]
      -- If not found in stopwords
      if firstin(filt, word) == 0 then
	 table.insert(words, word)
      end
   end
   if rev then
      words = revtab(words)
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
