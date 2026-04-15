# COMP 432 Project Package

## Project title
Fee-Capped Transaction Landing Prediction on Solana Blockchain: Learning a Low-Latency Sender Policy

## Files
- `COMP432_Solana_Landing_Project.ipynb` — main submission notebook (to be submitted on Moodle)
- `solana_execution_quality.py` — unified helper module (data collection, preparation, and modeling utilities)
- `demo_transactions.csv` — synthetic dataset so the notebook runs end-to-end immediately
- `requirements.txt` — minimal package list

## How to use

### 1. Setup
Put all files in the same working directory or ensure the helper module is importable.

### 2. Run in demo mode
Open `COMP432_Solana_Landing_Project.ipynb` in Colab or Jupyter and run all cells.

This uses `demo_transactions.csv` and verifies the full pipeline.

### 3. Generate real dataset (optional but required for final results)

#### Create a wallet (no CLI required)
python solana_execution_quality.py generate-keypair --output ~/.config/solana/id.json

#### Collect real transaction data
python solana_execution_quality.py collect-real-data --rpc-url https://api.devnet.solana.com --keypair ~/.config/solana/id.json --output real_transactions.csv --num-samples 300 --lamports 1 --fee-cap-lamports 20 --sleep-seconds 1.0

#### Prepare dataset for notebook
python solana_execution_quality.py prepare-real-data --input real_transactions.csv --output real_transactions_prepared.csv

Rename if needed:
mv real_transactions_prepared.csv real_transactions.csv

### 4. Run notebook with real data
- Set `USE_DEMO_DATA = False`
- Rerun all cells
- Update discussion and conclusion using real results

## Real-data note
The notebook is fully runnable using synthetic data. However, final academically valid conclusions should be based on real transaction data collected via the provided pipeline.

## Suggested submission workflow
- Verify the notebook using demo data
- Generate a real dataset using the helper module
- Rerun all notebook cells using real data
- Update discussion and conclusion sections
- Submit the notebook (`.ipynb`) as required

## Reproducibility note
The project is designed so that:
- the notebook demonstrates the full ML pipeline
- the helper module enables reproducible real-data collection and preparation

All experiments can be reproduced using the provided commands and configuration.
