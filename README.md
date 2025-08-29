# Video-LLaVA Backend

---

## Requirements

- Python >= 3.10
- Miniconda or Anaconda
- NVIDIA GPU with CUDA support
- CUDA drivers and toolkit installed (CUDA 12.8)

---

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/reqhiem/ask-llava.git
   cd ask-llava
```

2. Create and activate a virtual environment:

```bash
   conda create -n videollava python=3.10 -y
   conda activate videollava
```

3. Install dependencies:

```bash
   pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   pip install git+https://github.com/huggingface/transformers.git
```

---

## Generating requirements.txt from requirements.in

If you want to regenerate frozen dependencies:

```bash
   pip install pip-tools
   pip-compile requirements.in
   pip install -r requirements.txt
```

---

## Running the Backend

Once the installation is complete, you can start the backend service by running:

```bash
   python main.py
```
