--
require 'image'
--

--
return function()
	--
	local folder_pos = "folder-pos-trn"
	local folder_neg = "folder-neg-trn"
	local nchannels = 3
	local prob = 0.33

	--
	local bags_pos = {}
	local bags_neg = {}

	--
	-- generate positive bag table: ~33% of data
	--
	local p_pos = io.popen('ls ' .. folder_pos .. '/*.jpg')
	for path in p_pos:lines() do
		--
		-- discard bag with probability (1-prob)
		if math.random()<=prob then
			--

			-- load image data
			local data = image.load(path, nchannels, 'byte')

			if 1==nchannels then
				--
				if 3==data:size():size() then
					data = data:view(data:size(2)/data:size(3), data:size(3)*data:size(3))
				else
					data = data:view(data:size(1)/data:size(2), data:size(2)*data:size(2))
				end
			else
				--
				data = data:view(3, data:size(2)/data:size(3), data:size(3)*data:size(3)):transpose(1, 2):contiguous()
				data = data:view(data:size(1), 3*data:size(3))
			end

			-- insert bag into pos bagtable
			local bag = {}
			bag.data = data
			--bag.label = label
			table.insert(bags_pos, bag)
		end
	end
	p_pos:close()

	--
	-- generate negatives bag table: Random sampling until len pos (requires large neg set)
	--
	local p_neg = io.popen('ls ' .. folder_neg .. '/*.jpg')
	local count = #bags_pos
	while count > 0 do
		
		--randomly select a negative
		local path = p_neg[(math.random(1, #p_neg)]

		-- load image data
		local data = image.load(path, nchannels, 'byte')

		if 1==nchannels then
			--
			if 3==data:size():size() then
				data = data:view(data:size(2)/data:size(3), data:size(3)*data:size(3))
			else
				data = data:view(data:size(1)/data:size(2), data:size(2)*data:size(2))
			end
		else
			--
			data = data:view(3, data:size(2)/data:size(3), data:size(3)*data:size(3)):transpose(1, 2):contiguous()
			data = data:view(data:size(1), 3*data:size(3))
		end

		-- insert bag into pos bagtable
		local bag = {}
		bag.data = data
		--bag.label = label
		table.insert(bags_neg, bag)
		count = count - 1
	end 
	p_neg:close()
	
	--shuffle positives
	local n = math.min(n, #bags_pos)
	local p = torch.randperm(#bags_pos)  --array of random num length of bags_pos
	local bags_pos_shuffled = {}

	for i=1, n do
		--
		local bag = {}

		--bag.label = bags[ p[i] ].label
		bag.data = bags[ p[i] ].data
		bags_pos_shuffled[1+#bags_pos_shuffled] = bag
	end

	--generate triplets
	local triplets = {}
	local i
	local end_cond = #bags_pos_shuffled - 1

	for i=1, end_cond do
		
		table.insert(triplets, {bags_pos_shuffled[i].data, bags_pos_shuffled[i+1].data, bags_neg[i].data})

	end

	--return triplets
	return triplets
end