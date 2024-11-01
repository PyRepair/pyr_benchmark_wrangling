import os
import subprocess
from diff_utils import diff_utils
from datasets import load_dataset

dataset = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')

output_csv = "combined_localisation_measures.csv"

with open(output_csv, 'w') as outfile:
    outfile.write("") 

for i in range(len(dataset)):
    patch = dataset[i]['patch']
    
    # temporary diff
    diff_file = f"patch_{i}.diff"
    
    with open(diff_file, 'w') as f:
        f.write(patch)
    
    # script + capturing output
    command = f"python3.10 diff_utils.py --measure {diff_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # adding all the results to the same file 
    with open(output_csv, 'a') as outfile:
        outfile.write(result.stdout)  
        outfile.write("\n------------------\n")  
    
    os.remove(diff_file)

    print(f"processed patch {i} and appended output to {output_csv}")