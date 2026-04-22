# User Manual — Review Pulse Dashboard
### For Non-Technical Users

---

## What is Review Pulse?

Review Pulse is a web application that automatically reads customer reviews and tells you whether each review is **Positive**, **Negative**, or **Neutral**. You can analyze one review at a time, or upload a spreadsheet (CSV) of hundreds of reviews and get results instantly.

---

## Getting Started

### Step 1: Make sure the application is running

Ask your system administrator to run:
```
docker compose up -d
```

### Step 2: Open the dashboard

Open your web browser (Chrome, Firefox, Edge) and go to:
**http://localhost:3002**

You will see the Review Pulse dashboard with four tabs at the top.

---

## Tab 1: Single Review Analysis

Use this to quickly check the sentiment of one review.

**How to use:**
1. Click on the **"Single Review"** tab
2. Click inside the text box
3. Type or paste a customer review
4. Click the purple **"Analyze Sentiment"** button

**What you will see:**
- A coloured badge showing the result:
  - 🟢 **Positive** — the customer is happy
  - 🔴 **Negative** — the customer is unhappy
  - 🔵 **Neutral** — the review is mixed or unclear
- A **confidence bar** showing how certain the AI is (higher = more certain)
- Percentage scores for each category

> **Tip:** You can also press **Ctrl + Enter** on your keyboard to analyze quickly.

**Example:**
- Input: *"The delivery was fast and the product exceeded my expectations!"*
- Result: 🟢 Positive — 91% confidence

---

## Tab 2: Bulk CSV Upload

Use this to analyze many reviews at once from a spreadsheet file.

### Preparing your CSV file

Your file must be saved as a **.csv** file and must have a column with the header **review** (lowercase).

Example of correct format:
```
review
"This product is excellent!"
"Terrible quality, broke after one day"
"Delivery was okay, nothing special"
"Absolutely love it, will buy again"
```

You can create this in Microsoft Excel:
1. Open Excel → Column A header = **review**
2. Paste your reviews in column A rows 2 onwards
3. File → Save As → choose **CSV (Comma delimited)** format

### Uploading and analyzing

1. Click the **"Bulk CSV Upload"** tab
2. Click the upload area (or drag and drop your CSV file onto it)
3. Your file name will appear in purple when selected
4. Click the **"Analyze CSV"** button
5. Wait a few seconds for the results

### What you will see

**Summary cards** at the top showing:
- Total number of reviews analyzed
- How many were Positive, Negative, and Neutral

**Results table** showing each review with:
- The review text (shortened if long)
- The sentiment badge
- The confidence percentage

**Download button** — click **"⬇ Download Results CSV"** to save the results as a new spreadsheet file you can open in Excel.

---

## Tab 3: Dashboard (Charts and Insights)

After uploading a CSV, click this tab to see visual charts.

**What you will see:**

| Chart | What it shows |
|---|---|
| **Sentiment Distribution** | A donut chart showing the % of Positive, Negative, and Neutral reviews |
| **Confidence Distribution** | A bar chart showing how confident the AI was across all predictions |
| **Summary Stats** | Total reviews, positive rate %, negative rate %, average confidence |
| **Top Keywords** | The most common words found in Positive reviews vs Negative reviews |

**How to read the keyword section:**
Words that appear more often in positive reviews are shown in the green "Positive Keywords" section. Words that appear more in negative reviews are in the red "Negative Keywords" section. The number next to each word shows how many times it appeared.

---

## Tab 4: ML Pipeline

This tab shows the technical architecture of the system for reference. It includes:
- The data processing pipeline steps
- All available API endpoints
- The technology stack used

You do not need to interact with this tab during normal use.

---

## Status Indicator (Top Right Corner)

The coloured dot in the top right corner tells you the system status:

| Dot colour | Meaning |
|---|---|
| 🟢 Green — "Model Ready" | Everything is working normally |
| 🟡 Yellow — "Model Loading..." | The system is starting up, wait 30 seconds |
| 🔴 Red — "API Offline" | The backend is not running |

---

## Troubleshooting

| Problem | What to do |
|---|---|
| Page won't load | Check the URL is exactly **http://localhost:3002** |
| "API Offline" shown | Contact your administrator to restart the application |
| "Model Loading..." for more than 2 minutes | Refresh the page |
| CSV upload gives an error | Check your file has a column named exactly **review** |
| Charts don't show | Make sure you uploaded a CSV file first, then click Dashboard tab |
| Results seem wrong | Very short reviews (1-2 words) may be marked Neutral due to low confidence |

---

## Frequently Asked Questions

**Q: How accurate is the sentiment analysis?**
A: The AI model achieves approximately 83% accuracy on standard test data. Complex or sarcastic reviews may occasionally be misclassified.

**Q: What is "confidence"?**
A: Confidence shows how certain the AI is about its prediction. 90%+ means very confident. Below 65% is shown as Neutral.

**Q: How many reviews can I upload at once?**
A: Up to 1000 reviews per CSV upload.

**Q: Is my data stored anywhere?**
A: No. Your reviews are processed in memory and not saved to any database.

**Q: What languages are supported?**
A: Currently English only. The model was trained on English reviews.