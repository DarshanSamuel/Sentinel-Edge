import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/theme/app_theme.dart';
import '../providers/auth_provider.dart';
import '../widgets/glass_container.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  bool _isLogin = true;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _handleEmailSubmit() {
    final auth = context.read<AuthProvider>();
    if (_isLogin) {
      auth.loginWithEmail(_emailController.text, _passwordController.text);
    } else {
      auth.registerWithEmail(_emailController.text, _passwordController.text);
    }
  }

  void _handleGoogleSignIn() {
    final auth = context.read<AuthProvider>();
    auth.signInWithGoogle();
  }

  @override
  Widget build(BuildContext context) {
    final authProvider = context.watch<AuthProvider>();
    final isLoading = authProvider.isLoading;

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: AppTheme.backgroundGradient,
        ),
        child: Stack(
          children: [
            Positioned.fill(
              child: Opacity(
                opacity: 0.15,
                child: Image.asset(
                  'assets/images/background.png',
                  fit: BoxFit.cover,
                ),
              ),
            ),
            Center(
              child: GlassContainer(
                width: 400,
                padding: const EdgeInsets.all(32.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.security, color: AppTheme.accentNeonCyan, size: 60),
                    const SizedBox(height: 16),
                    Text(
                      'SENTINEL EDGE AI',
                      style: Theme.of(context).textTheme.displayLarge?.copyWith(fontSize: 24),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _isLogin ? 'SECURE LOGIN' : 'OPERATOR REGISTRATION',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        letterSpacing: 4,
                        color: AppTheme.accentNeonBlue,
                      ),
                    ),
                    const SizedBox(height: 40),
                    TextField(
                      controller: _emailController,
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        labelText: 'Operator ID (Email)',
                        labelStyle: const TextStyle(color: Colors.white54),
                        enabledBorder: OutlineInputBorder(
                          borderSide: BorderSide(color: Colors.white.withOpacity(0.3)),
                        ),
                        focusedBorder: const OutlineInputBorder(
                          borderSide: BorderSide(color: AppTheme.accentNeonCyan),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    TextField(
                      controller: _passwordController,
                      style: const TextStyle(color: Colors.white),
                      obscureText: true,
                      decoration: InputDecoration(
                        labelText: 'Passcode',
                        labelStyle: const TextStyle(color: Colors.white54),
                        enabledBorder: OutlineInputBorder(
                          borderSide: BorderSide(color: Colors.white.withOpacity(0.3)),
                        ),
                        focusedBorder: const OutlineInputBorder(
                          borderSide: BorderSide(color: AppTheme.accentNeonCyan),
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    
                    // Remember Me Checkbox
                    Row(
                      children: [
                        Checkbox(
                          value: authProvider.rememberMe,
                          onChanged: (val) {
                            if (val != null) {
                              context.read<AuthProvider>().rememberMe = val;
                            }
                          },
                          fillColor: MaterialStateProperty.resolveWith((states) {
                            if (states.contains(MaterialState.selected)) {
                              return AppTheme.accentNeonCyan;
                            }
                            return Colors.white24;
                          }),
                        ),
                        const Text('Remember Me', style: TextStyle(color: Colors.white70)),
                      ],
                    ),
                    
                    const SizedBox(height: 20),
                    // Show error message if any
                    if (authProvider.errorMessage != null)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 16),
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: AppTheme.alertRed.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: AppTheme.alertRed.withOpacity(0.4)),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.error_outline, color: AppTheme.alertRed, size: 20),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  authProvider.errorMessage!,
                                  style: const TextStyle(color: AppTheme.alertRed, fontSize: 13),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    
                    if (isLoading)
                      const CircularProgressIndicator(color: AppTheme.accentNeonCyan)
                    else
                      Column(
                        children: [
                          ElevatedButton(
                            onPressed: _handleEmailSubmit,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.accentNeonCyan.withOpacity(0.2),
                              foregroundColor: AppTheme.accentNeonCyan,
                              minimumSize: const Size(double.infinity, 50),
                              side: const BorderSide(color: AppTheme.accentNeonCyan),
                            ),
                            child: Text(_isLogin ? 'AUTHENTICATE' : 'REGISTER'),
                          ),
                          const SizedBox(height: 16),
                          const Text('OR', style: TextStyle(color: Colors.white54)),
                          const SizedBox(height: 16),
                          ElevatedButton.icon(
                            onPressed: _handleGoogleSignIn,
                            icon: const FaIcon(FontAwesomeIcons.google, color: Colors.white),
                            label: Text(_isLogin ? 'Sign in with Google' : 'Sign up with Google'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white.withOpacity(0.1),
                              foregroundColor: Colors.white,
                              minimumSize: const Size(double.infinity, 50),
                            ),
                          ),
                          const SizedBox(height: 24),
                          TextButton(
                            onPressed: () {
                              setState(() {
                                _isLogin = !_isLogin;
                                authProvider.errorMessage; // clear optionally
                              });
                            },
                            child: Text(
                              _isLogin ? "Don't have an account? Sign Up" : "Already an operator? Log In",
                              style: const TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                          ),
                        ],
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
