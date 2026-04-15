# COMP 432 Project 

## Project title
Fee-Capped Transaction Landing Prediction on Solana Blockchain: Learning a Low-Latency Sender Policy

## How to use

### 1. Setup
Put all files in the same working directory or ensure the helper module is importable.

### 2. Generate real dataset (optional)

#### Create a wallet (no CLI required)
python solana_execution_quality.py generate-keypair --output ~/.config/solana/id.json

#### Collect real transaction data
python solana_execution_quality.py collect-real-data --rpc-url https://api.mainnet-beta.solana.com --keypair ~/.config/solana/id.json --output real_transactions.csv --num-samples 300 --lamports 1 --fee-cap-lamports 20 --sleep-seconds 1.0

#### Prepare dataset for notebook
python solana_execution_quality.py prepare-real-data --input real_transactions.csv --output real_transactions_prepared.csv

Rename if needed:
mv real_transactions_prepared.csv real_transactions.csv

### 3. Run notebook with real data
- Rerun all cells
- Update discussion and conclusion using real results

## Reproducibility note
The project is designed so that:
- the notebook demonstrates the full ML pipeline
- the helper module enables reproducible real-data collection and preparation

All experiments can be reproduced using the provided commands and configuration.
