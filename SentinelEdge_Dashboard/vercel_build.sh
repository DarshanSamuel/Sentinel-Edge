#!/bin/bash
# Vercel Build Script for Flutter Web

# 1. Download Flutter SDK (Stable Channel)
git clone https://github.com/flutter/flutter.git -b stable --depth 1

# 2. Add flutter to PATH
export PATH="$PATH:`pwd`/flutter/bin"

# 3. Enable Web
flutter config --enable-web

# 4. Get Dependencies
flutter pub get

# 5. Build Web Project
flutter build web --release \
  --dart-define=FIREBASE_API_KEY=$FIREBASE_API_KEY \
  --dart-define=FIREBASE_APP_ID=$FIREBASE_APP_ID \
  --dart-define=FIREBASE_MESSAGING_SENDER_ID=$FIREBASE_MESSAGING_SENDER_ID \
  --dart-define=FIREBASE_PROJECT_ID=$FIREBASE_PROJECT_ID \
  --dart-define=FIREBASE_AUTH_DOMAIN=$FIREBASE_AUTH_DOMAIN \
  --dart-define=FIREBASE_STORAGE_BUCKET=$FIREBASE_STORAGE_BUCKET \
  --dart-define=FIREBASE_MEASUREMENT_ID=$FIREBASE_MEASUREMENT_ID
