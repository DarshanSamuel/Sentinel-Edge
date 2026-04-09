class TelemetryData {
  final DateTime timestamp;
  final double flowRate;
  final double pressure;
  final int modbusRegisterValue;
  final String modbusCommand;
  final bool isAnomalous;

  // Enriched metadata from SentinelEdge pipeline
  final String? classification;   // SAFE | SUSPICIOUS | THREAT
  final double? confidence;       // 0.0 - 1.0
  final String? reasoning;        // AI reasoning text
  final String? registerName;     // e.g. "chlorine_pump_speed"
  final String? functionCode;     // e.g. "FC06"
  final String? sourceIp;         // e.g. "192.168.1.22"
  final String? entryId;          // e.g. "SE-00010"
  
  // 21 Plant Parameters
  final Map<String, dynamic> plantState;

  TelemetryData({
    required this.timestamp,
    required this.flowRate,
    required this.pressure,
    required this.modbusRegisterValue,
    required this.modbusCommand,
    required this.isAnomalous,
    this.classification,
    this.confidence,
    this.reasoning,
    this.registerName,
    this.functionCode,
    this.sourceIp,
    this.entryId,
    this.plantState = const {},
  });
}
