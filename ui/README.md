# QURE Streamlit UI

Interactive dashboard for viewing reconciliation cases and agent decisions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at http://localhost:8501

## Features

### Dashboard
- Summary metrics (total cases, auto-approved, human review, rejected)
- Decision distribution chart
- Match score distribution
- Recent cases preview

### Case List
- Filter by decision type
- Filter by score range
- Sort by various criteria
- Expandable case details

### Case Details
- Full transaction details (GL and Bank)
- Agent scores visualization
- Policy decision explanation
- Action items

### Agent Performance
- Multi-agent architecture overview
- Accuracy metrics for each agent
- Decision statistics

### About
- System overview
- Architecture description
- Technology stack
- Non-negotiables

## Data

The UI loads test data from `data/synthetic/finance/`:
- `gl_transactions.json` - GL transactions
- `bank_transactions.json` - Bank transactions
- `expected_matches.json` - Expected match results

To regenerate test data:
```bash
cd ../..
python data/synthetic/generate_finance_data.py
```

## Screenshots

### Dashboard
Shows summary metrics and distribution charts.

### Case List
Browse and filter all reconciliation cases.

### Case Details
Deep dive into individual cases with agent scores and policy decisions.

## Future Enhancements

- [ ] Real-time case processing
- [ ] Agent execution visualization
- [ ] Audit trail viewer
- [ ] User management
- [ ] Export functionality
- [ ] Advanced filtering and search
