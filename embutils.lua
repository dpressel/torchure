-- http://lua-users.org/wiki/StringTrim
function trim(s)
  return s:match'^%s*(.*%S)' or ''
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
