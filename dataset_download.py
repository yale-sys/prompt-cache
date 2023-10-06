from datasets import load_dataset

## I will simply list up datasets below

print("Multi News\n", load_dataset('multi_news')[0])

print("MS_Marco", load_dataset('ms_marco', 'v2.1')[0])
