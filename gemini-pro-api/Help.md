## Help document for the Test.py or medgemma to test

requirements for CPU device
```text
accelerate
pillow
transformers
torch
huggingface-hub
requests
bitsandbytes
```
```bash
python -m venv venv # if not available
venv\Scripts\Activate.ps1
pip install -r requiremts.txt
python test.py --image-file chest_xray.png --hf-token <hf_token value>
```