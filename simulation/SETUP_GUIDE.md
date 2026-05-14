# Multi-Laptop Demo Setup Guide

This guide explains how to set up the multi-laptop simulation for the CogniShield AI-NGFW.

## Architecture

* **Laptop A (Attacker)**: Runs `multi_laptop_demo.py` to generate traffic.
* **Laptop B (Gateway/NGFW)**: Runs the FastAPI backend, React dashboard, and the Cowrie Honeypot Docker container.
* **Laptop C (Victim)**: Runs a simple HTTP server to simulate the target asset.

## Step-by-Step Setup

### 1. Laptop C (Victim)
1. Get the IP address of this machine (e.g., `192.168.1.103`).
2. Run a simple Python web server:
   ```bash
   python -m http.server 80
   ```

### 2. Laptop B (Gateway / NGFW)
1. Get the IP address of this machine (e.g., `192.168.1.102`).
2. Start the Honeypot:
   ```bash
   cd simulation
   docker-compose up -d
   ```
3. Start the FastAPI Service:
   ```bash
   cd ..
   uvicorn app.api.main:app --host 0.0.0.0 --port 8000
   ```
4. Start the React Frontend:
   ```bash
   cd app/frontend
   npm run dev
   ```

### 3. Laptop A (Attacker)
1. Ensure Python 3 is installed.
2. Run the simulation script, pointing to Laptop B and Laptop C:
   ```bash
   python simulation/multi_laptop_demo.py --gateway-ip 192.168.1.102 --victim-ip 192.168.1.103
   ```

## What to Observe
1. **Normal Traffic**: The script will send normal HTTP requests to Laptop C. On the React dashboard, you should see `ALLOW` decisions.
2. **Attack Traffic**: The script will switch to high-frequency SSH connections to Laptop B (simulating lateral movement).
3. **Detection**: The GNN will spike the `spatial_risk_score`, and the `EnsembleDetector` will flag the flows. The policy engine will issue a `BLOCK` decision.
4. **Redirection**: The `ActiveHoneypotBackend` will execute an `iptables` rule (or log it on Windows) redirecting the attacker's IP to the Cowrie honeypot on port 2222.
