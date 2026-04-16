# ⚡ QUICK START GUIDE

## Real-Time Medical Insurance Fraud Detection System
**By Bharath Kumar**

---

## 🚀 3-Step Setup

### Step 1: Install Dependencies (2 minutes)

Open PowerShell/Terminal in project directory:

```powershell
cd backend
pip install -r requirements.txt
```

### Step 2: Train Models (1-2 minutes)

```powershell
python model/train.py
```

Wait for "TRAINING COMPLETED SUCCESSFULLY!" message.

### Step 3: Start Application

**Terminal 1 - Start Backend:**
```powershell
python app.py
```

**Terminal 2 - Open Frontend:**
Double-click: `frontend/index.html`

OR visit: `file:///C:/Users/bhara/OneDrive/Documents/Desktop/project/frontend/index.html`

---

## ✅ Verify It's Working

1. **Backend Running**: See "Uvicorn running on http://0.0.0.0:8000" in terminal
2. **Frontend Loaded**: Browser shows "Medical Insurance Fraud Detection" page
3. **API Connected**: No error messages in browser console (F12)

---

## 🎯 Test It Out

### Quick Test 1: Legitimate Claim
Fill form with:
- Age: 35
- Gender: Female
- Claim Amount: 5000
- Hospital Stay: 3 days
- Previous Claims: 1
- Treatment Type: Outpatient

Click "Analyze Claim" → Should predict: **GENUINE**

### Quick Test 2: Fraudulent Claim
Fill form with:
- Age: 28
- Claim Amount: 48000
- Hospital Stay: 1 day
- Previous Claims: 15
- Treatment Type: Emergency
- Policy Age: 15 days

Click "Analyze Claim" → Should predict: **FRAUD**

---

## 🔍 What to Show in Demo

1. **Fill the form** with different values
2. **Click Analyze** - see real-time prediction
3. **View results**:
   - Fraud/Genuine badge
   - Risk score meter
   - XGBoost & Isolation Forest outputs
   - SHAP explanation chart
   - Top contributing features
4. **Open Admin Dashboard** - see statistics

---

## 📊 API Testing (Optional)

Visit: http://localhost:8000/docs

Try the interactive API documentation!

---

## ❌ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Run: `pip install -r backend/requirements.txt` |
| "Models not loaded" | Run: `python backend/model/train.py` |
| "Port 8000 in use" | Kill other processes or change port |
| Frontend not connecting | Check backend is running on port 8000 |

---

## 🎓 For Your Presentation

### Key Points to Highlight:

1. **Hybrid ML Approach**: Combines XGBoost (supervised) + Isolation Forest (unsupervised)
2. **Explainable AI**: Uses SHAP values to explain predictions
3. **Real-time**: Instant fraud detection (< 2 seconds)
4. **Production-Ready**: REST API, error handling, validation
5. **User-Friendly**: Modern dashboard interface

### Demo Flow:
1. Show the clean web interface
2. Enter a claim (use pre-filled values)
3. Click Analyze → Show loading state
4. Explain the results:
   - Main prediction
   - Risk score
   - Model outputs
   - Feature contributions (SHAP)
5. Show admin dashboard
6. Show API documentation

---

## 📁 Project Structure Overview

```
project/
├── backend/
│   ├── app.py              ← FastAPI server
│   ├── model/
│   │   ├── train.py        ← Train models
│   │   ├── predict.py      ← Make predictions
│   │   ├── hybrid_logic.py ← Combine models
│   │   └── shap_explainer.py ← Explain predictions
│   └── requirements.txt
│
└── frontend/
    ├── index.html          ← Main page
    ├── css/style.css       ← Styling
    └── js/app.js           ← Logic
```

---

## 💡 Pro Tips

- Keep backend terminal open while using frontend
- Use F12 in browser to see console logs
- Try different claim values to see different predictions
- Admin dashboard refreshes when you open it
- Models are already trained and saved after first run

---

## 🎉 You're All Set!

Your complete fraud detection system is ready to demo.

**Good luck with your presentation! 🚀**

---

**Need Help?** Check the full README.md for detailed documentation.
