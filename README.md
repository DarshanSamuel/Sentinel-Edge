# Sentinel-Edge: SCADA Telemetry Pipeline & AI Dashboard

This repository contains the full architecture for an advanced Industrial IoT cybersecurity evaluation pipeline. It combines an edge-deployed Modbus parsing / AI inference script (using local Large Language Models for classification) with a real-time reactive Flutter Dashboard that visually monitors ICS commands and enforces Role-Based Access Control (RBAC).

🔗 **[Live Vercel MVP Deployment](https://sentinel-edge-dashboard.vercel.app)**  
*(Note: As this is a Hackathon MVP, please allow a few seconds for the initial connection stream to establish).*

## 🏗️ Architectural Overview

![SentinelEdge Architecture & Process Flow](./SentinelEdge_FlowBlock_Diagram.png)

## 📁 Repository Structure

- `SentinelEdge_Pipeline/`: Python codebase simulating the embedded SCADA edge parser. Extracts mock Modbus data from the dataset, evaluates it using a local GGML/GGUF model via `llama.cpp`, and uploads structured results to Firebase via the Admin SDK.
- `SentinelEdge_Dashboard/`: The Flutter frontend (Dart/Google UI). Acts as the primary operations dashboard with color-coded Threat parsing, live Modbus graphs (via fl_chart), and strict user-registration boundaries.
- `Colab_Scripts/`: Contains the Jupyter notebooks utilized for fine-tuning the base `Gemma-2-2b-it` model on the custom SCADA dataset.

## 🚀 Setup & Execution Guide

### 1. Flutter Dashboard Setup
The Dashboard enforces strict RBAC (Role-Based Access Control). Newly generated accounts are immediately blocked until an Admin approves them.
1. Download Flutter SDK and resolve dependencies: `flutter pub get`
2. Navigate to `lib/firebase_options.dart` and enter your Firebase web configurations.
3. Run the application: `flutter run -d web`
4. Register a new account.
5. Go to your Firebase Cloud Firestore Console, find the newly generated user in the `users` collection, and manually change their fields to `role: 'admin'` and `status: 'approved'`. Refersh the Flutter app and login!

### 2. Vercel MVP Deployment (Flutter Web)
This project is configured for secure, CI/CD automated deployments utilizing Vercel. All sensitive Firebase credentials have been abstracted into Environment Variables to strictly comply with Hackathon public-repository rules.

1. Create a New Project in Vercel and import this GitHub repository.
2. Under **Project Settings** during import, you must specify the **Root Directory** as `SentinelEdge_Dashboard`. (This is critical because it is a monorepo).
3. The Build Command and Output Directory are automatically handled by the included `vercel.json` and `vercel_build.sh` scripts.
4. Expand the **Environment Variables** section.
5. Simply copy the contents of your local `.env` file (which should have been securely provided to you) and paste them into the first Vercel Key input field. Vercel will automatically parse and bulk-load all 7 keys (e.g., `FIREBASE_API_KEY`, `FIREBASE_PROJECT_ID`).
6. Click **Deploy**. Vercel will securely inject your credentials during the `--dart-define` compilation step without exposing them.

### 3. Edge Python Pipeline Setup
This service acts as the physical layer data injector.
1. Go to your Firebase project and generate a new Service Account Private Key.
2. Rename the json to `service-account.json` and place it in the pipeline directory.
3. Install dependencies: `pip install firebase-admin`
4. Run the simulation loop: `python firebase_upload_service.py --continuous --delay 3` 

## 🛡️ License & Disclaimers
This repository is heavily tailored around educational IIoT research. Production deployments must integrate strict firewall layers around the `service-account.json`.
