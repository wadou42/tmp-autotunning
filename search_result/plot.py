import re
import matplotlib.pyplot as plt
from itertools import accumulate

def plot_multi(data: list[list[float]], labels:list[str], title: str = "Plot") -> None:
    """
    Plots multiple lines on a single graph.

    Parameters:
    - data: A list of lists, where each inner list contains y-values for a line plot.
    - title: Title of the plot.
    """

    new_data = [list(accumulate(data_line, max)) for data_line in data]
    data = new_data
    plt.figure()
    
    for i, y_values in enumerate(data):
        x_values = list(range(len(y_values)))
        plt.plot(x_values, y_values, label=labels[i] if i < len(labels) else f'Line {i+1}')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()



# 18: O3_perf=457.58623674802215, opt_perf=443.5022379167353, acc_rate=0.9692211047880741
def parser_autotuning_data(file_path: str) -> list[float]:
    """
    Parses a file to extract floating-point numbers.

    Parameters:
    - file_path: Path to the file to be parsed.

    Returns:
    - A list of floating-point numbers extracted from the file.
    """
    float_pattern = re.compile(
        r'\d+:\s*O3_perf=([-+]?(?:\d*\.\d+|\d+|inf)),\s*'
        r'opt_perf=([-+]?(?:\d*\.\d+|\d+|inf)),\s*'
        r'acc_rate=([-+]?(?:\d*\.\d+|\d+|inf))', 
        re.IGNORECASE
    )
    extracted_floats = []

    with open(file_path, 'r') as file:
        for line in file:
            matches = re.search(float_pattern, line)
            if matches:
                if matches.group(3) == '-1':
                    extracted_floats.append(float(-1))
                else:
                    extracted_floats.append(float(matches.group(3)))
                    

    return extracted_floats

def parser_srtuner_data(file_path: str) -> list[float]:
    """
    Parses a SRTuner log file to extract floating-point numbers.

    Parameters:
    - file_path: Path to the SRTuner log file.

    Returns:
    - A list of floating-point numbers extracted from the file.
    """
    """current trial: 1.041s, best performance so far: 1.041s"""
    float_pattern = re.compile(r'best performance so far:\s*([-+]?\d*\.\d+|\d+|inf)s')
    extracted_floats = []

    with open(file_path, 'r') as file:
        for line in file:
            matches = re.search(float_pattern, line)
            if matches:
                if matches.group(1) == 'inf':
                    print(matches.group(1))
                    extracted_floats.append(float(-1))
                else:
                    extracted_floats.append(1 / float(matches.group(1)))

    return extracted_floats

if __name__ == "__main__":
    sf_list = [ "redis", "scann", "doris"]
    for sf in sf_list:
        ir2vec_path = f"ir2vec.search/{sf}.SA/perf.recorder.txt"
        autotuning_path = f"search/{sf}.SA/perf.recorder.txt"
        srtuner_path = f"SRTuner_LOG/{sf}_srtuner.log"
        ir2vec_data = parser_autotuning_data(ir2vec_path)
        autotuning_data = parser_autotuning_data(autotuning_path)
        srtuner_data = parser_srtuner_data(srtuner_path)
        plot_multi([ir2vec_data, autotuning_data, srtuner_data], labels=["IR2Vec", "AutoTuning", "SRTuner"], title=f"Tunning {sf} Performance Comparison")
        