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
flutter build web --release
