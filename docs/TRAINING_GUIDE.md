# Training on Colab & Kaggle with Zenodo Data

Since your dataset is hosted on Zenodo, you can download it directly into your cloud environments (Google Colab or Kaggle Kernels) using the command line.

## 1. Google Colab

Google Colab provides a temporary environment. You will need to download and unzip the dataset at the start of each session.

**Copy and paste this into the first cell of your notebook:**

```python
# 1. Install utility to download from Zenodo
!pip install -q zenodo-get

# 2. Download the dataset (Record ID: 18214758)
print("Downloading dataset... (approx 10GB)")
!zenodo_get 18214758

# 3. Unzip the dataset
print("Unzipping... this may take 2-5 minutes")
!unzip -q agri_foundation_v1.zip -d ./dataset

print("Done! Dataset is ready at ./dataset/release/data")
```

### Loading the Data (PyTorch Example)

```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

data_dir = "./dataset/release/data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Loaded {len(dataset)} images from {len(dataset.classes)} classes.")
```

---

## 2. Kaggle Notebooks

You have two options on Kaggle.

### Option A: Direct Download (Temporary)
Use the same code as the Google Colab section above. This downloads the data every time you run the notebook.

### Option B: Create a Persistent Kaggle Dataset (Recommended)
This is better because you only download it once, and then you can attach it to any notebook instantly.

1.  Open a new **Kaggle Notebook**.
2.  Run the download/unzip code (Option A).
3.  On the right sidebar, go to **Settings** -> **Persistence** -> set to **"Files only"** or **"Variables and Files"**.
    *   *Note: This might not work perfectly for 10GB output depending on Kaggle's limits.*
4.  **BETTER METHOD: Import via URL**
    1.  Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
    2.  Click **"New Dataset"**.
    3.  Select **"Link to Remote File"**.
    4.  Enter the direct download link from your Zenodo record.
        *   *Go to your [Zenodo page](https://zenodo.org/records/18214758), right-click the "Download" button next to the zip file, and choose "Copy Link Address".*
    5.  Kaggle will download it in the background and create a dataset you can add to any notebook.
