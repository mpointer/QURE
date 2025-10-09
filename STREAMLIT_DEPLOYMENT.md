# QURE Streamlit Cloud Deployment Guide

## Files Ready ✅

- `requirements.txt` - All Python dependencies
- `.streamlit/config.toml` - Theme and configuration
- `ui/streamlit_app.py` - Main application
- `data/synthetic/` - All demo data files (5 verticals)

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `QURE`
3. Description: "QURE - Quality Unified Resolution Engine"
4. Visibility: **Public** (required for free Streamlit Cloud)
5. **DON'T** initialize with README/gitignore/license
6. Click "Create repository"

## Step 2: Push to GitHub

After creating the repo, run these commands:

```bash
cd C:\Users\micha\Documents\GitHub\QURE
git remote add origin https://github.com/YOUR_USERNAME/QURE.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Configure:
   - **Repository**: `YOUR_USERNAME/QURE`
   - **Branch**: `main`
   - **Main file path**: `ui/streamlit_app.py`
5. Click "Deploy!"

## Your Live URL

After deployment (2-3 minutes), your app will be at:
`https://YOUR_USERNAME-qure.streamlit.app`

## What's Included

### 5 Business Verticals with Demo Data:
1. **Finance** - GL/Bank Reconciliation (20 cases)
2. **Healthcare** - Prior Authorization (20 cases)
3. **Insurance** - Subrogation Claims (20 cases)
4. **Retail** - Inventory Reconciliation (20 cases)
5. **Manufacturing** - PO/Receipt Matching (20 cases)

### Features:
- ✅ ROI Calculator showing cost/time savings
- ✅ Animated business impact metrics
- ✅ Live processing with QRU workflow visualization
- ✅ Decision reasoning chains
- ✅ Manual vs QURE comparison charts
- ✅ Multi-vertical navigation
- ✅ Audit trail visualization
- ✅ Agent performance dashboard

## Troubleshooting

If deployment fails, check:
1. Main file path is exactly: `ui/streamlit_app.py`
2. Repository is public
3. All files pushed successfully (`git status` shows clean)

## Local Testing

To test locally before deploying:
```bash
cd C:\Users\micha\Documents\GitHub\QURE
streamlit run ui/streamlit_app.py
```

## Support

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Issues: Create an issue in your GitHub repo
