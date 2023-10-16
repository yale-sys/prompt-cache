from datasets import load_dataset

## I will simply list up datasets below

datasets = {}


def load_documentation_summary():
    ## Summary dataset
    datasets['multi_news'] = load_dataset('multi_news')
    return datasets['multi_news']
    # print("Multi News\n", datasets['multi_news']['train'][0])


def load_multidoc_qna():
    ## Open domain question answering
    # = version 2.1 =
    # datasets['ms_marco'] = load_dataset('ms_marco', 'v2.1')
    # print("MS_Marco", datasets['ms_marco']['train'][0])

    # = version 1.1 =
    datasets['ms_marco'] = load_dataset('ms_marco', 'v1.1')
    # print("MS_Marco", datasets['ms_marco']['train'][0])
    return datasets['ms_marco']


pass
