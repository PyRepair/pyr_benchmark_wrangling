import codecs
import json
import os
import time

program_dir = "../QuixBugs/python_programs"
test_dir = "../QuixBugs/python_testcases"

program_files = os.listdir(program_dir)

patch_count = 0
total = 0

for program_file in program_files:
    if program_file.endswith(".py"):
        total += 1

        test_file = os.path.join(test_dir, "test_" + program_file)

        for line in range(1, 50):
            command = (
                "python3 pyrepair/cli.py -t naive -p"
                f" {os.path.join(program_dir, program_file)} -ts"
                f" {test_file} -l {line} --outform json"
            )

            start_time = time.time()
            output = os.popen(command).read()
            execution_time = time.time() - start_time
            try:
                output_data = json.loads(output)
            except json.decoder.JSONDecodeError:
                if "Invalid file path: QuixBugs/python_testcases/" in output:
                    break

            if output_data.get("patch_found"):
                patch_count += 1
                break
            if output_data.get("error"):
                print(output_data.get("error"))


print(f"Number of patches found: {patch_count}\nTotal programs: {total}")
