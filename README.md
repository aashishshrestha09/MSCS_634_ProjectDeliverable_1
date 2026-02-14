# Advanced Data Mining for Data-Driven Insights and Predictive Modeling

## Deliverable 1: Data Collection, Cleaning, and Exploration

|                 |                                                               |
| :-------------- | :------------------------------------------------------------ |
| **Course**      | MSCS 634 — Advanced Big Data and Data Mining (Spring 2026)    |
| **Institution** | University of the Cumberlands                                 |
| **Team**        | Mazen Abdul Rahman Mohammed · Aashish Shrestha · Mahesh Gaire |
| **Date**        | February 13, 2026                                             |

---

## 📋 Dataset Summary

| Property        | Value                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------- |
| **Dataset**     | [Online Retail](https://www.kaggle.com/datasets/vijayuv/onlineretail)                        |
| **Source**      | Kaggle (originally from UCI Machine Learning Repository)                                     |
| **Domain**      | E-Commerce / Sales Transactions                                                              |
| **Raw Records** | 541,909                                                                                      |
| **Attributes**  | 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) |
| **Time Period** | December 2010 – December 2011                                                                |
| **Description** | Transactional data for a UK-based online retailer specializing in unique all-occasion gifts  |

### Why This Dataset?

- **Size:** 541K+ records and 8 attributes exceed the minimum requirement (500+ records, 8–10 attributes).
- **Data Quality Issues:** Contains missing values, duplicates, cancelled transactions, and noisy data — providing ample opportunity to demonstrate cleaning techniques.
- **Future Suitability:** The transactional structure supports regression (revenue prediction), classification (customer segmentation), clustering (RFM analysis with K-Means), and association rule mining (Market Basket Analysis with Apriori).

---

## 🧹 Data Cleaning Steps

| Step | Action                                     | Records Affected | Justification                                                                                                                       |
| ---- | ------------------------------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Remove missing `CustomerID`**            | ~135,080 rows    | CustomerID is essential for customer-level analysis (clustering, classification). Imputation is not feasible for identifier fields. |
| 2    | **Remove missing `Description`**           | ~1,454 rows      | Product descriptions are needed for association rule mining in Deliverable 4.                                                       |
| 3    | **Remove duplicate rows**                  | ~5,225 rows      | Exact duplicates are data entry errors and inflate statistics.                                                                      |
| 4    | **Remove cancelled transactions**          | ~8,905 rows      | Invoices starting with 'C' are reversed transactions, not actual sales.                                                             |
| 5    | **Remove non-positive Quantity/UnitPrice** | ~40 rows         | Zero or negative values represent errors or non-standard entries.                                                                   |
| 6    | **Convert data types**                     | All rows         | `InvoiceDate` → datetime; `CustomerID` → integer for consistency.                                                                   |
| 7    | **Feature engineering**                    | All rows         | Created `TotalPrice = Quantity × UnitPrice` for revenue analysis.                                                                   |

**Result:** The cleaned dataset contains approximately **392,692 rows** (9 columns) with zero missing values.

---

## 🔍 Key Insights from Exploratory Data Analysis

1. **Right-Skewed Distributions:** Quantity, UnitPrice, and TotalPrice are all heavily right-skewed (skewness > 10), indicating that most transactions involve small quantities at low prices, with a long tail of high-value orders. _Implication:_ Log-transformation will be needed for regression modeling.

2. **Significant Outliers:** IQR-based analysis reveals thousands of outliers in each numerical feature, representing bulk/wholesale orders. _Implication:_ Capping at the 99th percentile or robust scaling will be applied before modeling.

3. **Seasonal Revenue Trends:** Revenue shows a strong upward trend from summer to autumn, peaking in November 2011 (holiday shopping season). _Implication:_ Month/quarter features will improve predictive accuracy.

4. **UK Dominance:** The United Kingdom accounts for approximately 82% of total revenue across 38 countries. _Implication:_ Geographic stratification and one-hot encoding of Country will be important for modeling.

5. **Pareto Customer Distribution:** A small fraction of customers generates disproportionate revenue, following the 80/20 rule. _Implication:_ This distribution is ideal for K-Means customer segmentation using RFM analysis.

6. **Two Purchasing Behaviors:** Scatter plot analysis reveals two distinct patterns — bulk buyers (high quantity, low price) and premium buyers (low quantity, high price). _Implication:_ These natural clusters support classification modeling.

7. **Temporal Patterns:** No sales occur on Saturdays; peak hours are 10 AM–3 PM; Thursday generates the highest revenue. _Implication:_ Day-of-week and hour features can be engineered for improved predictions.

---

## ⚠️ Challenges Encountered and Solutions

| Challenge                                                 | Solution                                                                                                                                             |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **~25% of records lack CustomerID**                       | Removed these rows since CustomerID is essential for customer-level analysis and cannot be meaningfully imputed for an identifier field.             |
| **Cancelled transactions mixed with valid sales**         | Identified cancellations by the 'C' prefix in InvoiceNo and filtered them out to ensure only genuine sales are analyzed.                             |
| **Heavy outliers distorting distributions**               | Used IQR-based detection and clipped visualizations at the 99th percentile for readability. Full outlier treatment will be applied in Deliverable 2. |
| **Special characters in product descriptions**            | Used `encoding='ISO-8859-1'` when loading the CSV to handle non-ASCII characters in the Description field.                                           |
| **InvoiceDate stored as string**                          | Converted to `datetime64` to enable time-series analysis and extraction of temporal features (month, day, hour).                                     |
| **Multicollinearity (TotalPrice = Quantity × UnitPrice)** | Documented for awareness; Ridge Regression (L2 regularization) will be used in Deliverable 2 to mitigate this.                                       |

---

## 📁 Repository Structure

```
MSCS_634_ProjectDeliverable_1/
├── .gitignore               # Git ignore rules (excludes large CSV)
├── OnlineRetail.ipynb       # Jupyter Notebook with full analysis
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation (this file)
```

> **Note:** `OnlineRetail.csv` (~45 MB) is excluded from version control via `.gitignore` due to GitHub's file-size limits. See the **Download the dataset** step below to obtain it.

---

## ▶️ How to Run

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<username>/MSCS_634_ProjectDeliverable_1.git
   cd MSCS_634_ProjectDeliverable_1
   ```

2. **Download the dataset:**

   The CSV is not included in the repository due to GitHub's file-size limits. Download it from either source and place it in the project root directory:
   - **Kaggle:** [Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail) — click _Download_ and extract `OnlineRetail.csv`.
   - **UCI ML Repository:** [Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail) — download the Excel file, open it, and export as CSV.

   After downloading, your directory should contain:

   ```
   MSCS_634_ProjectDeliverable_1/
   ├── OnlineRetail.csv    ← place the file here
   ├── OnlineRetail.ipynb
   ├── requirements.txt
   └── ...
   ```

3. **Create and activate a virtual environment** _(recommended):_

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

4. **Install dependencies from `requirements.txt`:**

   ```bash
   pip install -r requirements.txt
   ```

   All required packages and their minimum versions are pinned in [`requirements.txt`](requirements.txt) to ensure reproducibility.

5. **Run the notebook:**
   ```bash
   jupyter notebook OnlineRetail.ipynb
   ```
   Alternatively, open `OnlineRetail.ipynb` in **VS Code** (with the Jupyter extension) and run all cells sequentially.

> **Note:** The `requirements.txt` file lists every dependency needed to reproduce the analysis. Using `pip install -r requirements.txt` ensures that all collaborators and reviewers work with the same package versions.
