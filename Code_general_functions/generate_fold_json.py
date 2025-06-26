import json

def create_folds_json(input_file, output_file):
    """
    Create a new JSON file containing subject IDs grouped by folds.

    :param input_file: Path to the input JSON file
    :param output_file: Path to the output JSON file
    
    :return: None
    """
    # Load the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
        training_data = data['training']

    print("training_data = ", training_data)

    # Initialize a dictionary to hold the folds
    folds = {
        "fold0": [],
        "fold1": [],
        "fold2": [],
        "fold3": []
    }

    # Group subject IDs by their folds
    for subject in training_data:
        print("subject = ", subject)
        subject_id = subject['subject_id']
        print("got unique subject_id = ", subject_id)
        fold = subject['fold']
        folds[f"fold{fold}"].append(subject_id)

    # Save the new JSON file
    with open(output_file, 'w') as f:
        json.dump(folds, f, indent=4)

# Example usage
input_file = '/home/linuxuser/user/data/Dataset001_ChoroidPlexus_T1_sym_AP/dataset_train_val_pred.json'
output_file = '/home/linuxuser/user/data/pazienti/folds.json'


create_folds_json(input_file, output_file)
print("Saving folds to JSON file at ", output_file)