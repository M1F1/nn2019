import torch
import utils_data
import numpy as np
import csv
import os
from pdb import set_trace as st
if __name__ == '__main__':
    model = torch.load('models/new.pt')
    model.eval()
    batch_size = 64
    data_dir = 'data'
    test_dataset = utils_data.Project1Dataset(data_dir='data',
                                              which='test')
    all_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    all_predictions = np.zeros((len(test_dataset),), dtype=np.int8)
    with torch.no_grad():
        # initialize the number of correct predictions
        correct: int = 0
        for i, (x, y) in enumerate(all_data_loader):
            # pass through the network
            output: torch.Tensor = model(x)
            # update the number of correctly predicted examples
            pred = output.max(1)[1]
            # st()
            print(F'Predicting: {i}', flush=False)
            all_predictions[i*batch_size:(i+1)*batch_size] = pred
    with open(os.path.join(data_dir, "sample_submission_new_1.csv"), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'label'])
        writer.writerows(enumerate(all_predictions))


