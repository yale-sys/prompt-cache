import os, requests, zipfile, io, sys
from tqdm import tqdm
from bleurt import score

BLEURT_20_URL = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
CHECKPOINT = "BLEURT-20"

def download_bleurt_20():
    if not os.path.exists('./BLEURT-20/'):
        print("Downloading BLEURT-20 checkpoint...")
        with requests.get(BLEURT_20_URL, stream=True) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kbyte
            buffer = io.BytesIO()
            # Initialize tqdm with the total file size, and use it as an iterable in a loop
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in response.iter_content(block_size):
                    buffer.write(data)
                    # Update tqdm about the downloaded data size
                    pbar.update(len(data))
                    
            buffer.seek(0)
            
            # Unzip the file
            print("Unzipping BLEURT-20 checkpoint...")
            with zipfile.ZipFile(buffer) as zip_file:
                # Extract all the contents into the current working directory
                zip_file.extractall(path=".")

class BleurtScorer:
    def __init__(self):
        download_bleurt_20()
        self.scorer = score.BleurtScorer(CHECKPOINT)

    def score(self, refs=[str], hyps=[str]):
        return self.scorer.score(references=refs, candidates=hyps)

if __name__ == '__main__':
    bs = BleurtScorer()
    print(bs.score(["i'm leo"], ["my name is leo"]))
