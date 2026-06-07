# Project Structure

```text
road-accident/

├── app/
│
│   ├── detector/
│
│   ├── tracker/
│   │   └── botsort_tracker.py
│
│   ├── trajectory/
│   │   ├── history.py
│   │   ├── velocity.py
│   │   ├── predictor.py
│   │   └── ttc.py
│
│   ├── lanes/
│   │   └── lanes.py
│
│   ├── risk/
│   │   ├── conflict_score.py
│   │   └── severity.py
│
│   ├── vizualization/
│   │   └── draw_boxes.py
│
│   └── main.py
│
├── outputs/
│
├── videos/
│   └── test.mp4
│
├── weights/
│   └── yolo11m.pt
│
├── requirements.txt
├── README.md
└── .gitignore
```

# Steps to Run

### 1. Clone Repository

```bash
git clone <repository-url>
cd road-accident
```

### 2. Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO11 Weights

Create:

```text
weights/
```

Place:

```text
weights/yolo11m.pt
```

### 5. Add Input Video

Place CCTV video in:

```text
videos/test.mp4
```

### 6. Run Project

```bash
python app/main.py
```

### 7. Stop Execution

Press:

```text
Q
```

### 8. Output

Processed video will be saved in:

```text
outputs/result.mp4
```
