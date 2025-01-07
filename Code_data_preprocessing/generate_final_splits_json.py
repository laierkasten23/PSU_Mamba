import json
import sys

def generate_final_splits(input_path, output_path):
    """
    Generate final splits json file from the input json file. 

    Args:
        input_path (_type_): _description_
        output_path (_type_): _description_
        
        Usage: python generate_final_splits_json.py <input_path> <output_path>
        e.g. python generate_final_splits_json.py "/mnt/turing/LIA/pazienti/folds.json" "/var/datasets/LIA/Umamba_data/nnUNet_preprocessed/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/splits_final.json"
        
    """    
    
    print("Reading data from", input_path, "and writing final splits to", output_path)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    folds = list(data.keys())
    print("Folds:", folds)
    final_splits = []

    for i in range(len(folds)):
        val_fold = folds[i]
        train_folds = [fold for fold in folds if fold != val_fold]

        train_data = []
        for fold in train_folds:
            train_data.extend(data[fold])

        val_data = data[val_fold]

        final_splits.append({
            "train": train_data,
            "val": val_data
        })

    with open(output_path, 'w') as f:
        json.dump(final_splits, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_final_splits_json.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_final_splits(input_path, output_path)