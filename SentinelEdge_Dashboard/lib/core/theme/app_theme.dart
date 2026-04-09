import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Colors
  static const Color primaryDark = Color(0xFF0D1321); // Deep dark blue
  static const Color secondaryDark = Color(0xFF000000); // Pure black
  static const Color accentNeonCyan = Color(0xFF00F0FF); // Neon Cyan for UI
  static const Color accentNeonBlue = Color(0xFF0A66C2);
  static const Color safeGreen = Color(0xFF39FF14); // Neon Green
  static const Color alertRed = Color(0xFFFF003C); // Neon Red

  // Gradient
  static const LinearGradient backgroundGradient = LinearGradient(
    colors: [primaryDark, secondaryDark],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: Colors.transparent,
      colorScheme: const ColorScheme.dark(
        primary: accentNeonCyan,
        secondary: accentNeonBlue,
        surface: Color(0xFF1E2A38),
        background: primaryDark,
      ),
      textTheme: GoogleFonts.orbitronTextTheme(
        ThemeData.dark().textTheme,
      ).copyWith(
        displayLarge: GoogleFonts.orbitron(fontSize: 32, fontWeight: FontWeight.bold, color: Colors.white),
        titleLarge: GoogleFonts.inter(fontSize: 20, fontWeight: FontWeight.w600, color: Colors.white),
        bodyMedium: GoogleFonts.inter(fontSize: 14, color: Colors.white70),
      ),
      cardTheme: CardThemeData(
        color: Colors.white.withOpacity(0.05),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
      ),
    );
  }
}
