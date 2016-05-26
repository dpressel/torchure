package = "torchure"
version = "scm-1"
source = {
  url = "git://github.com/dpressel/torchure.git",
}
description = {
  summary = "Simple tools for using existing Word2Vec data",
  detailed = [[
    Various torch utilities including support to load and save word2vec models
  ]],
  homepage = "https://github.com/dpressel/torchure",
  license = "MIT"
}
dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
}
build = {
  type = "builtin",
  modules = {
    ["torchure.init"] = "init.lua",
    ["torchure.utils"] = "utils.lua",
    ["torchure.Word2VecModel"] = "Word2VecModel.lua",
    ["torchure.Word2VecLookupTable"] = "Word2VecLookupTable.lua",
  }
}
