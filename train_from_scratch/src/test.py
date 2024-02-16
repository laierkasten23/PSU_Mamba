import argparse

import torch
import tqdm


if __name__ == '__main__':
    
    '''
    Test the model on unseen data.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', required=True)
    parser.add_argument('-p', '--path', help='Path for storing testing results', required=False, default='test_results')
    parser.add_argument('-d', '--data', help='Path to test dataset') # TODO: Add default path
    parser.add_argument('-b', '--batchsize', help='Test Batch Size', default=2, type=int)

    args = parser.parse_args()
    test_dir = args.data
    batch_size = args.batchsize

    path_all_model_files_root = f"{args.path}/{args.name}/"
    test_metrics_path = path_all_model_files_root + "test_metrics/"
    evaluation_images_path = path_all_model_files_root + "evaluation_images/"
    model_checkpoint_path = path_all_model_files_root + "training_checkpoints/"