import os
import numpy as np

from utils import softmax_1d

if __name__ == "__main__":
    architectures = ['template', 'pseudo_inverse']
    arch_comparison = []
    for arch in architectures:
        print(arch)
        fname = 'avi_' + arch + '.npy'
        traces = np.load(os.path.join('debug', fname))
        traces = traces.transpose(1, 0, 2)
        arch_comparison.append(traces)
    for i in range(5):
        for arch_traces in arch_comparison:
            softmaxed_latents = softmax_1d(np.mean(arch_traces[i], axis=0))
            print(softmaxed_latents)
            print(max(softmaxed_latents))
        print('\n')