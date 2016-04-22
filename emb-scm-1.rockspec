package = "emb"
version = "scm-1"
source = {
  url = "git://github.com/dpressel/emb.git",
}
description = {
  summary = "Simple tools for using existing Word2Vec data",
  detailed = [[
    Load and use word2vec models, and provide an easy-to-use LookupTable implementation
  ]],
  homepage = "https://github.com/dpressel/emb",
  license = "MIT"
}
dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
}
build = {
  type = "builtin",
  modules = {
    ["emb.init"] = "init.lua",
    ["emb.embutils"] = "embutils.lua",
    ["emb.Word2VecModel"] = "Word2VecModel.lua",
    ["emb.Word2VecLookupTable"] = "Word2VecLookupTable.lua",
  }
}
