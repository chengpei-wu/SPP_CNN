# pre-set parameters
# network parameters

# CNN parameters:
# training network sizes
training_network_scale = '(300,700)'
# training network attributes: isd: directed network; isw: weighted network
isd = 1
# training labels: yc: controllability robutness; lc: connectivity robustness
training_robustness = 'yc'
# testing network sizes
testing_network_scale = '(300,700)'


# label generating algorithm：
# ndeg: remove nodes by degree rank
# nrnd: random remove nodes
# nbet: remove nodes by betweeness centrality rank
if training_robustness == 'lc':
    attack_strategy = 'ndeg'
else:
    attack_strategy = 'nrnd'
isw = 0

# note that the batch_size must be fixed to 1
# unless the training network sizes are fixed
batch_size = 1
epochs = 30

# the valid set size
valid_proportion = 0.1
