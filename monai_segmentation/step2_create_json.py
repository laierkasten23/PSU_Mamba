import os
import json
import math
from collections import OrderedDict

# https://github.com/FAIR-Unipd/ASCHOPLEX/blob/main/Code/create_json.py
# and https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/notebooks/msd_datalist_generator.ipynb 

join = os.path.join

class WriteJSON:
    """
    Class for writing .json files to run finetuning and/or the prediction of Choroid Plexus segmentations.

    - If finetune is set to 'yes', the method prepares to include training data in the JSON file. 
    It reads image and label files from the image_Tr and label_Tr directories respectively, 
    checks if the number of images and labels match, and splits the data into training and validation sets.
    The splitting is done in such a way that the first half of the data is used for training and the second half for validation.

    - If prediction is set to 'yes', the method prepares to include testing data in the JSON file. It reads image files from the 
    image_Ts directory and includes them in the testing data.
            

    Args:
        dataroot (str): The root directory of the dataset. Default is ".".
        description (str, optional): Description of the dataset. Default is None.
        work_dir (str): The working directory. Default is ".".
        finetune (str): Flag indicating whether to perform finetuning. Default is ".".
        prediction (str): Flag indicating whether to perform prediction. Default is ".".
    """

    def __init__(self, dataroot: str=".", description=None, work_dir: str=".", finetune: str=".", prediction: str="."):
        """
        Initializes the WriteJSON object.
        """
        self.dataroot=dataroot
        if description is None:
            self.description='Dataset for Choroid Plexus segmentation'
        elif isinstance(description, str):
            self.description=description
        self.JSON_dir=work_dir
        self.finetune=finetune
        self.prediction=prediction
        self.file=[]
       
       
    def write_json_file(self):
        """
        Writes the .json file based on the provided parameters.

        Returns:
            str: The path of the generated .json file.
        """
        
        # set data path
        output_folder = join(self.JSON_dir, 'JSON_file')
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        if self.finetune=='yes':
            train_id=True

            if self.prediction=='yes':
                test_id=True
                test_ft=False

            else: 
                # self.prediction=='no'
                test_id=False
                test_ft=False
                
            name_json="dataset_finetuning.json"

        else:

            train_id=False

            if self.prediction=='yes':
                test_id=True
                test_ft=False
                name_json="dataset_prediction.json"

            else:
                # self.prediction=='ft'
                test_id=False
                test_ft=True
                name_json="dataset_finetuning.json"


        if train_id:
            train_dir = join(self.dataroot, 'image_Tr')
            label_dir = join(self.dataroot, 'label_Tr')
            train_ids=[]
            validation_ids=[]
            label_train_ids = []
            label_valid_ids=[]

            filenames_image = os.listdir(train_dir)
            filenames_image.sort()
            filenames_label = os.listdir(label_dir)
            filenames_label.sort()   

            if len(filenames_image)!=len(filenames_label):
                raise ValueError("The number of images and the number of labels is different. Please, check image_Tr and label_Tr folders.")

            # training
            # use 1/2 of the data for training and 1/2 for validation
            jj=math.ceil(len(filenames_image)/2)

            for name in filenames_image[0:jj]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(train_dir, name)
                train_ids.append(image)
            
            count=0 
            for name in filenames_label[0:jj]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                img_=os.path.basename(filenames_image[count]).replace('_image', '')
                lab_=os.path.basename(name).replace('_seg', '')
                if img_==lab_:
                    label=join(label_dir, name)
                    label_train_ids.append(label)
                    count+=1
                else:
                    raise ValueError("Subject identifier is not univoque. Please, pass correct data")
            
            # validation (the rest of the data)

            for name in filenames_image[jj:len(filenames_image)]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(train_dir, name)
                validation_ids.append(image)
            count=jj
            for name in filenames_label[jj:len(filenames_image)]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                img_=os.path.basename(filenames_image[count]).replace('_image', '')
                lab_=os.path.basename(name).replace('_seg', '')
                if img_==lab_:
                    label=join(label_dir, name)
                    label_valid_ids.append(label)
                    count+=1
                else:
                    raise ValueError("Subject identifier is not univoque. Please, pass correct data")


        if test_id or test_ft:
            #  testing
            
            test_dir=join(self.dataroot, 'image_Ts')
            test_ids=[]
            testnames = os.listdir(test_dir)
            testnames.sort()

            for test_name in testnames:
                if not(test_name.endswith('.nii') | test_name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(test_dir, test_name)
                test_ids.append(image)


        if (train_id and test_id) or (not(train_id) and test_id) or (train_id and not(test_id)):

            # create json file - manually set

            json_dict = OrderedDict()
            json_dict['name'] = "MRI Dataset - Choroid Plexus Segmentation" 
            json_dict['description'] = self.description
            json_dict['tensorImageSize'] = "3D"
            json_dict['modality'] = {
                "0": "MR"
            }
            
            json_dict['labels'] = {
                "0": "background",
                "1": "Choroid Plexus"

            }

            if train_id and test_id:

                json_dict['numTraining'] = len(train_ids)
                json_dict['numValidation'] = len(validation_ids)
                json_dict['numTest'] = len(test_ids)
                json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for j, i in zip(label_train_ids, train_ids)]
                json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for j,i in zip(label_valid_ids, validation_ids)]
                json_dict['testing'] = [{"image": '%s' %i} for i in test_ids]

            elif not(train_id) and test_id: 

                json_dict['numTest'] = len(test_ids)
                json_dict['testing'] = [{"image": '%s' % i} for i in test_ids]

            elif train_id and not(test_id):

                json_dict['numTraining'] = len(train_ids)
                json_dict['numValidation'] = len(validation_ids)
                json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for j, i in zip(label_train_ids, train_ids)]
                json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for j,i in zip(label_valid_ids, validation_ids)]

            with open(join(output_folder, name_json), 'w') as f: # opens file for writing and automatically close the file after block of code.
                json.dump(json_dict, f, indent=4, sort_keys=True) # writes json_dict dictionary to the file f in JSON format.
        
        elif test_ft:

            # append lines to the .json files
            with open(join(output_folder, name_json)) as f:
                json_append=json.load(f)

            json_append['numTest']=len(test_ids)

            json_append['testing'] = [{"image": '%s' % i} for i in test_ids]

            with open(join(output_folder, name_json), 'w') as f:
                json.dump(json_append, f, indent=4, sort_keys=True)

        self.file.append(join(output_folder, name_json))

        return self.file[0] 

