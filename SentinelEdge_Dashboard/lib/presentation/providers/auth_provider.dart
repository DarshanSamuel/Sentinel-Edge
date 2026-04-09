import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../../domain/models/user_model.dart';

class AuthProvider extends ChangeNotifier {
  bool _isAuthenticated = false;
  bool _isLoading = false;
  String? _errorMessage;
  bool _googleSignInInitialized = false;

  bool _rememberMe = true;
  UserModel? _currentUserModel;

  bool get isAuthenticated => _isAuthenticated;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  bool get rememberMe => _rememberMe;
  UserModel? get currentUserModel => _currentUserModel;

  set rememberMe(bool val) {
    _rememberMe = val;
    notifyListeners();
  }

  Future<void> _checkAndRegisterUser(User firebaseUser) async {
    final docRef = FirebaseFirestore.instance.collection('users').doc(firebaseUser.uid);
    final docSnap = await docRef.get();

    if (!docSnap.exists) {
      // First time sign up
      final newUser = UserModel(
        uid: firebaseUser.uid,
        email: firebaseUser.email ?? 'unknown@example.com',
        role: 'user', // default role
        status: 'pending', // awaits admin approval
      );
      await docRef.set(newUser.toMap());
      _currentUserModel = newUser;
    } else {
      _currentUserModel = UserModel.fromDocument(docSnap);
    }

    if (!_currentUserModel!.isApproved) {
      // Block entry and sign out locally
      await signOut();
      throw Exception('Your account is awaiting Admin approval.');
    }
  }

  Future<void> signInWithGoogle() async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      UserCredential userCredential;
      
      if (kIsWeb) {
        // WEB SOLUTION
        final googleProvider = GoogleAuthProvider();
        googleProvider.addScope('email');
        googleProvider.setCustomParameters({'prompt': 'select_account'});
        
        userCredential = await FirebaseAuth.instance.signInWithPopup(googleProvider);
      } else {
        // NATIVE SOLUTION
        if (!_googleSignInInitialized) {
          await GoogleSignIn.instance.initialize(clientId: null);
          _googleSignInInitialized = true;
        }

        final GoogleSignInAccount? googleUser = await GoogleSignIn.instance.authenticate();
        if (googleUser == null) {
          _isLoading = false;
          notifyListeners();
          return;
        }

        final GoogleSignInAuthentication auth = googleUser.authentication;
        final credential = GoogleAuthProvider.credential(
          idToken: auth.idToken,
        );
        userCredential = await FirebaseAuth.instance.signInWithCredential(credential);
      }

      // Check RBAC approval
      if (userCredential.user != null) {
        await _checkAndRegisterUser(userCredential.user!);
        
        // If we reach here, user is approved
        if (!_rememberMe && kIsWeb) {
          await FirebaseAuth.instance.setPersistence(Persistence.SESSION);
        }
        
        _isAuthenticated = true;
        _errorMessage = null;
      }
    } catch (e) {
      _errorMessage = '$e'.replaceAll('Exception: ', '');
      debugPrint(_errorMessage);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> registerWithEmail(String email, String password) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();
    try {
      final userCredential = await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      if (userCredential.user != null) {
        await _checkAndRegisterUser(userCredential.user!);
        if (!_rememberMe && kIsWeb) await FirebaseAuth.instance.setPersistence(Persistence.SESSION);
        _isAuthenticated = true;
      }
    } catch (e) {
      _errorMessage = '$e'.replaceAll('Exception: ', '');
      debugPrint(_errorMessage);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> loginWithEmail(String email, String password) async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();
    try {
      final userCredential = await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      if (userCredential.user != null) {
        await _checkAndRegisterUser(userCredential.user!);
        if (!_rememberMe && kIsWeb) await FirebaseAuth.instance.setPersistence(Persistence.SESSION);
        _isAuthenticated = true;
      }
    } catch (e) {
      _errorMessage = '$e'.replaceAll('Exception: ', '');
      debugPrint(_errorMessage);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> signOut() async {
    _isLoading = true;
    notifyListeners();
    try {
      await FirebaseAuth.instance.signOut();
      if (!kIsWeb) {
        await GoogleSignIn.instance.signOut();
      }
      _isAuthenticated = false;
      _currentUserModel = null;
    } catch (e) {
      debugPrint('Sign out error: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
