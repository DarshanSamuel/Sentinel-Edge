import 'package:cloud_firestore/cloud_firestore.dart';

class UserModel {
  final String uid;
  final String email;
  final String role; // 'admin' or 'user'
  final String status; // 'pending' or 'approved'

  UserModel({
    required this.uid,
    required this.email,
    required this.role,
    required this.status,
  });

  factory UserModel.fromDocument(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>? ?? {};
    return UserModel(
      uid: doc.id,
      email: data['email'] ?? '',
      role: data['role'] ?? 'user',
      status: data['status'] ?? 'pending',
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'email': email,
      'role': role,
      'status': status,
    };
  }

  bool get isAdmin => role == 'admin';
  bool get isApproved => status == 'approved';
}
