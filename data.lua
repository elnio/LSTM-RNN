--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

vocab_idx = 0
vocab_map = {}

--[[
This function takes a 1-D tensor of size (M) and produces a 2-D tensor of size (M/batch_size x batch_size).
So given:
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
and batch_size of 10 it produces:
1   3   5   7   9  11  13  15  17  19
2   4   6   8  10  12  14  16  18  20
--]]
-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     -- this sub line here populates the column (from 1-size in the first dimension talks about the rows,
     -- and i,i takes a single column.
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

--[[
This function reads a file with many sentences sequentially. It first replaces the new line
with <eos> and then creates a tensor (x), of size equal to the number of the words in the
file. It then populates this tensor with an index which is the index of the word in a hashmap
that is created on the fly as we parse the words.
So in the end x is a tensor which contains a unique index for each word that occurred in the
document.
--]]
local function load_data(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', '<eos>')
   data = stringx.split(data)
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
      end
      x[i] = vocab_map[data[i]]
   end
   return x
end

local function traindataset(batch_size)
   local x = load_data(ptb_path .. "ptb.train.txt")
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
   local x = load_data(ptb_path .. "ptb.test.txt")
   x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   return x
end

local function validdataset(batch_size)
   local x = load_data(ptb_path .. "ptb.valid.txt")
   x = replicate(x, batch_size)
   return x
end

--[[
So the datasets that we are going to be training/testing on are of size 
(words_in_document/batch_size x batch_size) and the word ordering follows the columns
(so 1st till batch_size-th word, lie in the first column, batch_size-th+1 till 2*batch_size-th
word lie on the second column and so forth.
--]]
return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset}
