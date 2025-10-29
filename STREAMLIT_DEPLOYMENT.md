# QURE Streamlit Cloud Deployment Guide

## Files Ready ✅

- `requirements.txt` - All Python dependencies
- `.streamlit/config.toml` - Theme and configuration
- `ui/streamlit_app.py` - Main application
- `data/synthetic/` - All demo data files (5 verticals)

## Step 1: GitHub Repository ✅

Repository already exists and is configured:
- **Repository**: https://github.com/mpointer/QURE
- **Branch**: `main`
- **Visibility**: Public
- **Latest Commit**: Phase 5 Complete (Planner QRU + UI Enhancements)

All code has been pushed to GitHub and is ready for deployment.

## Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub (use your mpointer account)
3. Click "New app"
4. Configure:
   - **Repository**: `mpointer/QURE`
   - **Branch**: `main`
   - **Main file path**: `ui/streamlit_app.py`
5. Click "Deploy!"

## Your Live URL

After deployment (2-3 minutes), your app will be at:
`https://mpointer-qure.streamlit.app`

Or Streamlit may assign:
`https://qure.streamlit.app` (if available)

## What's Included

### 5 Business Verticals with Demo Data:
1. **Finance** - GL/Bank Reconciliation (20 cases)
2. **Healthcare** - Prior Authorization (20 cases)
3. **Insurance** - Subrogation Claims (20 cases)
4. **Retail** - Inventory Reconciliation (20 cases)
5. **Manufacturing** - PO/Receipt Matching (20 cases)

### Executive Pages:
- ✅ **Executive Summary** - High-level business metrics and ROI
- ✅ **Before & After** - Manual vs QURE comparison
- ✅ **Business Case** - Financial justification and TCO analysis
- ✅ **What-If Scenarios** - Interactive scenario planning

### Technical Pages:
- ✅ **Dashboard** - Real-time metrics across all verticals
- ✅ **Use Cases** - Vertical-specific demo scenarios
- ✅ **Live Processing** - Animated QRU workflow with Planner intelligence
- ✅ **Case List** - Searchable case inventory
- ✅ **Case Details** - Deep-dive into individual decisions
- ✅ **Audit Trail** - Complete decision history
- ✅ **QRU Performance** - Agent effectiveness metrics

### Phase 5 Features (NEW):
- 🧠 **Planner QRU** - Intelligent meta-orchestrator with business domain expertise
- 📊 **16 Business Problem Classifications** across 5 verticals
- 💰 **87.7% Cost Savings** on simple cases via intelligent QRU selection
- ⏭️ **Visual Skip Indicators** - Clear display of skipped vs executed QRUs
- 🎯 **Dynamic Workflow Generation** - Execution plans tailored to case complexity

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
