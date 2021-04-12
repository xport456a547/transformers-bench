Objectif: faire des steps de 2^18 (262144)

Pour seq = 2048
sur v100 32Go : 
	Roberta Base : 1 GPU batch 2 max
	262144 = 64/n_gpu * 2 * 2048
	=> définir la taille des batch à 64/n_gpu
	batch_size = 8 pour 8 GPU 
