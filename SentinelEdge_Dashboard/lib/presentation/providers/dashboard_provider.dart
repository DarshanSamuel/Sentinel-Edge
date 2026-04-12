import 'dart:async';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../../domain/models/telemetry_data.dart';

class DashboardProvider extends ChangeNotifier {
  final List<TelemetryData> _logs = [];
  List<TelemetryData> get logs => List.unmodifiable(_logs);

  bool _isSystemSafe = true;
  bool get isSystemSafe => _isSystemSafe;

  bool _isPlantActive = true;
  bool get isPlantActive => _isPlantActive;
  
  int _consecutiveAnomalies = 0;

  StreamSubscription? _firestoreSubscription;
  StreamSubscription? _killSwitchSubscription;

  DashboardProvider() {
    _startFirebaseStream();
    _listenToKillSwitch();
  }

  void _listenToKillSwitch() {
    _killSwitchSubscription = FirebaseFirestore.instance
        .collection('system_status')
        .doc('kill_switch')
        .snapshots()
        .listen((doc) {
      if (doc.exists && doc.data() != null) {
        _isPlantActive = doc.data()!['is_active'] as bool? ?? true;
        notifyListeners();
      }
    });
  }

  void _startFirebaseStream() {
    _firestoreSubscription = FirebaseFirestore.instance
        .collection('scada_telemetry')
        .orderBy('timestamp', descending: true)
        .limit(200) // Ensure we get enough history for scatter charts
        .snapshots()
        .listen((snapshot) {
      _logs.clear();
      for (var doc in snapshot.docs) {
        final data = doc.data();
        _logs.add(TelemetryData(
          timestamp: data['timestamp'] != null
              ? (data['timestamp'] as Timestamp).toDate()
              : DateTime.now(),
          flowRate: (data['flow_rate'] as num?)?.toDouble() ?? 0.0,
          pressure: (data['pressure'] as num?)?.toDouble() ?? 0.0,
          modbusRegisterValue: data['modbus_register_value'] as int? ?? 0,
          modbusCommand: data['modbus_code'] as String? ?? 'UNKNOWN',
          isAnomalous: data['is_anomalous'] as bool? ?? false,
          classification: data['classification'] as String?,
          confidence: (data['confidence'] as num?)?.toDouble(),
          reasoning: data['reasoning'] as String?,
          registerName: data['register_name'] as String?,
          functionCode: data['function_code'] as String?,
          sourceIp: data['source_ip'] as String?,
          entryId: data['entry_id'] as String?,
          plantState: data['plant_state'] != null ? Map<String, dynamic>.from(data['plant_state']) : {},
        ));
      }
      
      // Compute System Safety and Kill Switch Logic
      if (_logs.isNotEmpty) {
        // Calculate consecutive anomalies based on the most recent logs first
        // Since logs are descending (newest at index 0)
        int anomalies = 0;
        for (var log in _logs) {
          if (log.isAnomalous) {
            anomalies++;
          } else {
            break; // Stop counting once we hit a SAFE log
          }
        }
        
        _consecutiveAnomalies = anomalies;
        
        // Auto-kill plant if 5 consecutive anomalies
        if (_consecutiveAnomalies >= 5 && _isPlantActive) {
          killPlant(); // Emergency kill switch triggered globally
        }

        if (_logs.first.isAnomalous) {
          _isSystemSafe = false;
          // Only auto-reset safe status if plant hasn't been killed
          if (_isPlantActive) {
            Timer(const Duration(seconds: 5), () {
              // Ensure another check hasn't proved it unsafe again
              if (_logs.isNotEmpty && !_logs.first.isAnomalous) {
                _isSystemSafe = true;
                notifyListeners();
              }
            });
          }
        } else {
          _isSystemSafe = true;
        }
      } else {
        _isSystemSafe = true;
      }

      notifyListeners();
    });
  }

  void killPlant() {
    // Write unconditionally to Firestore to trigger true IoT halt
    FirebaseFirestore.instance.collection('system_status').doc('kill_switch').set({'is_active': false});
  }

  void resumePlant() {
    _consecutiveAnomalies = 0; // Reset
    _isSystemSafe = true;
    FirebaseFirestore.instance.collection('system_status').doc('kill_switch').set({'is_active': true});
  }

  @override
  void dispose() {
    _firestoreSubscription?.cancel();
    _killSwitchSubscription?.cancel();
    super.dispose();
  }
}

