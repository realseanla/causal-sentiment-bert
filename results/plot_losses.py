#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Sentiment Causal BERT')
	parser.add_argument('path', metavar='PATH', type=str, help='input data')
	parser.add_argument('output_dir', metavar='DIR', type=str, help='output dir')

	args = parser.parse_args()

	train = {'total': [], 'train_g': [], 'train_Q': [], 'mlm': []}
	validation = {'validation_g': [], 'validation_Q': []}
	
	with open(args.path, 'r') as f:
		for line in f:
			tokens = line.split(' ')	
			if 'train' in tokens and "Epoch" in tokens:
				loss = float(tokens[-1])
				if 'total' in tokens:
					train['total'].append(loss)	
				elif 'propensity' in tokens:
					train['train_g'].append(loss)	
				elif 'conditional' in tokens:
					train['train_Q'].append(loss)
				elif 'masked' in tokens:
					train['mlm'].append(loss)
			elif 'dev' in tokens and "Epoch" in tokens:
				loss = float(tokens[-1])
				if 'propensity' in tokens:
					validation['validation_g'].append(loss)	
				elif 'conditional' in tokens:
					validation['validation_Q'].append(loss)
	train_losses = pd.DataFrame.from_dict(train)
	train_losses.loc[:, 'epoch'] = train_losses.index.astype(int) + 1
	validation_losses = pd.DataFrame.from_dict(validation)

	losses = pd.concat([train_losses, validation_losses], axis=1)

	filename = os.path.splitext(args.path)[0]
	output_path = "{}/{}.png".format(args.output_dir, filename)

	fig = losses.plot(x='epoch', 
                          y=['total', 'mlm', 'train_g', 'validation_g', 'train_Q', 'validation_Q'], 
                          title='Training and Validation losses over epochs').get_figure()
	fig.savefig(output_path)
