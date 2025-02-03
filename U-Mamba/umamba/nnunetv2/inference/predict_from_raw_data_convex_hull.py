import os
import torch
from os.path import join
from typing import Optional, List, Tuple, Union
from nnunetv2.training.data_augmentation.custom_transforms import ConvexHullTransform
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class nnUNetPredictorWithConvexHull(nnUNetPredictor):
    def __init__(self, plans_manager, dataset_json, model_training_output_dir, use_folds, checkpoint_name):
        super().__init__(plans_manager, dataset_json, model_training_output_dir, use_folds, checkpoint_name)
        # Initialize Convex Hull Transform
        convex_hull_path = os.path.join(model_training_output_dir, 'convex_hull.npy')
        self.convex_hull_transform = ConvexHullTransform(convex_hull_path=convex_hull_path)

    def predict_from_data_iterator(self, data_iterator, output_folder, save_probabilities=False, overwrite=True):
        """
        This function applies the Convex Hull transformation to the data before making predictions.
        """
        for data, properties in data_iterator:
            # Apply Convex Hull Transform to the test data
            data_dict = {'data': data, 'seg': None, 'properties': properties}
            data_dict = self.convex_hull_transform(**data_dict)
            data = data_dict['data']

            # Perform prediction
            self.network.eval()
            with torch.no_grad():
                prediction = self.network(data)

            # Save the prediction
            self.save_prediction(prediction, properties, output_folder, save_probabilities, overwrite)
            
    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    # print('sleeping')
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                    #                               dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
                    #                                                             configuration_manager, label_manager,
                    #                                                             properties,
                    #                                                             save_probabilities)
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret