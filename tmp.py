import re

"""
2: O3_perf=2417.0, opt_perf=2614.0, acc_rate=0.9246365723029839
3: O3_perf=2417.0, opt_perf=2495.0, acc_rate=0.9687374749498998
4: O3_perf=2417.0, opt_perf=2591.0, acc_rate=0.9328444615978386
5: O3_perf=2417.0, opt_perf=2511.0, acc_rate=0.9625647152528873
"""

def modify_file(file_path, old_O3_perf, new_O3_perf):
    with open(file_path, 'r') as file:
        content = file.readlines()

    updated_lines = []
    for line in content:
        # Match the pattern for O3_perf, opt_perf, and acc_rate
        match = re.match(r'(\d+): O3_perf=(\d+\.?\d*), opt_perf=(\d+\.?\d*), acc_rate=(\d+\.?\d*)', line)
        if match:
            line_num, O3_perf, opt_perf, acc_rate = match.groups()
            O3_perf = float(O3_perf)
            opt_perf = float(opt_perf)

            # Replace O3_perf with the new value if it matches the old value
            if O3_perf == old_O3_perf:
                O3_perf = new_O3_perf

            # Recalculate acc_rate
            acc_rate = opt_perf / O3_perf

            # Reconstruct the line with updated values
            updated_line = f"{line_num}: O3_perf={O3_perf}, opt_perf={opt_perf}, acc_rate={acc_rate:.16f}\n"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)  # Keep lines that don't match the pattern

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)
