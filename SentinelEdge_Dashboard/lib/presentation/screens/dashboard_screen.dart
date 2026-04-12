import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import '../providers/dashboard_provider.dart';
import '../providers/auth_provider.dart';
import '../widgets/glass_container.dart';
import 'package:intl/intl.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  // Map Modbus codes to numeric Y-axis values for the chart
  static const Map<String, int> _modbusCodeToIndex = {
    'MB001': 1, 'MB002': 2, 'MB003': 3, 'MB004': 4, 'MB005': 5,
    'MB006': 6, 'MB007': 7, 'MB008': 8, 'MB009': 9, 'MB010': 10,
    'MB011': 11, 'MB012': 12, 'MB013': 13, 'MB014': 14, 'MB015': 15,
    'MB016': 16, 'MB017': 17, 'MB018': 18, 'MB019': 19, 'MB020': 20,
    'MB021': 21, 'MB022': 22, 'MB023': 23, 'MB024': 24, 'MB025': 25,
    'MB026': 26, 'MB027': 27, 'MB028': 28,
  };

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: AppTheme.backgroundGradient,
          ),
          child: Stack(
            clipBehavior: Clip.none,
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
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildHeader(context),
                      const SizedBox(height: 20),
                      TabBar(
                        indicatorColor: AppTheme.accentNeonCyan,
                        labelColor: AppTheme.accentNeonCyan,
                        unselectedLabelColor: Colors.white54,
                        tabs: const [
                          Tab(text: 'OVERVIEW & TELEMETRY', icon: Icon(Icons.dashboard)),
                          Tab(text: 'PLANT PARAMETERS', icon: Icon(Icons.settings_input_component)),
                          Tab(text: 'TREND GRAPHS', icon: Icon(Icons.show_chart)),
                        ],
                      ),
                      const SizedBox(height: 20),
                      Expanded(
                        child: TabBarView(
                          children: [
                            _buildOverviewTab(context),
                            _buildParametersTab(context),
                            _buildTrendGraphsTab(context),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    final user = context.watch<AuthProvider>().currentUserModel;
    
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          crossAxisAlignment: WrapCrossAlignment.center,
          spacing: 16,
          runSpacing: 16,
          children: [
            const Icon(Icons.security, color: AppTheme.accentNeonCyan, size: 40),
            Text(
              'Sentinel Edge AI',
              style: Theme.of(context).textTheme.displayLarge?.copyWith(fontSize: 24),
            ),
          ],
        ),
        Wrap(
          spacing: 16,
          runSpacing: 16,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            if (user != null && user.isAdmin) ...[
              ElevatedButton.icon(
                onPressed: () => _showAdminPanel(context),
                icon: const Icon(Icons.admin_panel_settings, color: Colors.white),
                label: const Text('Admin Panel', style: TextStyle(color: Colors.white)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.accentNeonBlue.withOpacity(0.4),
                  side: const BorderSide(color: AppTheme.accentNeonBlue),
                ),
              ),
            ],
            IconButton(
              icon: const Icon(Icons.logout, color: Colors.white70),
              onPressed: () {
                context.read<AuthProvider>().signOut();
              },
              tooltip: 'Logout',
            ),
            _buildFloatingKillSwitch(),
          ],
        ),
      ],
    );
  }

  void _showAdminPanel(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: const Color(0xFF161B22), // GitHub Dark style
          title: const Text('Admin Control Panel: Pending Operators', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          content: SizedBox(
            width: 600,
            height: 400,
            child: StreamBuilder<QuerySnapshot>(
              stream: FirebaseFirestore.instance.collection('users').where('status', isEqualTo: 'pending').snapshots(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }
                if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                  return const Center(child: Text('No pending users.', style: TextStyle(color: Colors.white54)));
                }

                return ListView.builder(
                  itemCount: snapshot.data!.docs.length,
                  itemBuilder: (context, index) {
                    final doc = snapshot.data!.docs[index];
                    final data = doc.data() as Map<String, dynamic>;
                    
                    return Card(
                      color: Colors.white10,
                      child: ListTile(
                        leading: const Icon(Icons.person_outline, color: Colors.amber),
                        title: Text(data['email'] ?? 'Unknown', style: const TextStyle(color: Colors.white)),
                        subtitle: Text('Role: ${data['role']} | Status: ${data['status']}', style: const TextStyle(color: Colors.white54)),
                        trailing: ElevatedButton(
                          onPressed: () async {
                            await doc.reference.update({'status': 'approved'});
                          },
                          style: ElevatedButton.styleFrom(backgroundColor: AppTheme.safeGreen),
                          child: const Text('APPROVE', style: TextStyle(color: Colors.white)),
                        ),
                      ),
                    );
                  },
                );
              },
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Close', style: TextStyle(color: AppTheme.accentNeonCyan)),
            ),
          ],
        );
      },
    );
  }

  Widget _buildFloatingKillSwitch() {
    return Consumer<DashboardProvider>(
// .... REST OF KILL SWITCH UNCHANGED ....
      builder: (context, provider, child) {
        if (!provider.isPlantActive) {
          return GlassContainer(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.warning, color: AppTheme.alertRed, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'PLANT KILLED',
                  style: TextStyle(color: AppTheme.alertRed, fontWeight: FontWeight.bold, fontSize: 12),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: () => provider.resumePlant(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.safeGreen,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  ),
                  child: const Text('RESUME', style: TextStyle(color: Colors.white, fontSize: 12)),
                )
              ],
            ),
          );
        }

        return GlassContainer(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: provider.isSystemSafe ? AppTheme.safeGreen : Colors.amber,
                  boxShadow: [
                    BoxShadow(
                      color: (provider.isSystemSafe ? AppTheme.safeGreen : Colors.amber).withOpacity(0.6),
                      blurRadius: 8,
                      spreadRadius: 2,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Text(
                provider.isSystemSafe ? 'SYSTEM SECURE' : 'ANOMALY DETECTED',
                style: TextStyle(
                  color: provider.isSystemSafe ? AppTheme.safeGreen : Colors.amber,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: () => provider.killPlant(),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent,
                  side: const BorderSide(color: AppTheme.alertRed),
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                ),
                child: const Text('KILL SWITCH', style: TextStyle(color: AppTheme.alertRed, fontSize: 12)),
              )
            ],
          ),
        );
      },
    );
  }

  Widget _buildOverviewTab(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 900;

    if (isMobile) {
      return SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildSystemStatus(context),
            const SizedBox(height: 20),
            SizedBox(height: 400, child: _buildModbusCommandChart(context)),
            const SizedBox(height: 20),
            SizedBox(height: 400, child: _buildModbusLogs(context)),
          ],
        ),
      );
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          flex: 2,
          child: Column(
            children: [
              _buildSystemStatus(context),
              const SizedBox(height: 20),
              Expanded(child: _buildModbusCommandChart(context)),
            ],
          ),
        ),
        const SizedBox(width: 24),
        Expanded(
          flex: 1,
          child: _buildModbusLogs(context),
        ),
      ],
    );
  }

  Widget _buildSystemStatus(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 900;
    return Consumer<DashboardProvider>(
      builder: (context, provider, child) {
        if (provider.logs.isEmpty) {
          return GlassContainer(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.cloud_off, color: Colors.white38, size: 24),
                const SizedBox(width: 12),
                Text(
                  'Waiting for telemetry from SentinelEdge...',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.white38),
                ),
              ],
            ),
          );
        }
        final latest = provider.logs.first;
        final classLabel = latest.classification ?? (latest.isAnomalous ? 'THREAT' : 'SAFE');
        final classColor = classLabel == 'SAFE'
            ? AppTheme.safeGreen
            : classLabel == 'SUSPICIOUS'
                ? Colors.amber
                : AppTheme.alertRed;
        
        final displayLabel = classLabel == 'SUSPICIOUS' ? 'SUSP' : classLabel;

        Widget commandBox = GlassContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('CURRENT COMMAND', style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 8),
              Text(
                latest.modbusCommand,
                style: Theme.of(context).textTheme.displayLarge?.copyWith(color: AppTheme.accentNeonCyan, fontSize: 36),
              ),
            ],
          ),
        );

        Widget detailsBox = GlassContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('COMMAND DETAILS', style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 8),
              Text(
                latest.registerName ?? 'Unknown Register',
                style: Theme.of(context).textTheme.displayLarge?.copyWith(color: AppTheme.accentNeonBlue, fontSize: 24),
              ),
            ],
          ),
        );

        Widget inferenceBox = GlassContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('INFERENCE', style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 8),
              Row(
                children: [
                  Container(
                    width: 14,
                    height: 14,
                    decoration: BoxDecoration(shape: BoxShape.circle, color: classColor),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    displayLabel,
                    style: Theme.of(context).textTheme.displayLarge?.copyWith(color: classColor, fontSize: 28),
                  ),
                ],
              ),
              if (latest.confidence != null)
                Text(
                  'Confidence Tracker: ${(latest.confidence! * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(color: Colors.white54, fontSize: 13, fontWeight: FontWeight.bold)
                ),
            ],
          ),
        );

        if (isMobile) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              commandBox,
              const SizedBox(height: 12),
              detailsBox,
              const SizedBox(height: 12),
              inferenceBox,
            ],
          );
        }

        return Row(
          children: [
            Expanded(child: commandBox),
            const SizedBox(width: 20),
            Expanded(flex: 2, child: detailsBox),
            const SizedBox(width: 20),
            Expanded(child: inferenceBox),
          ],
        );
      },
    );
  }

  /// Modbus Command (code) vs Timestamp scatter chart
  Widget _buildModbusCommandChart(BuildContext context) {
    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('MODBUS TIMELINE', style: Theme.of(context).textTheme.titleLarge),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _legendDot(AppTheme.safeGreen, 'SAFE'),
                  const SizedBox(width: 16),
                  _legendDot(Colors.amber, 'SUSPICIOUS'),
                  const SizedBox(width: 16),
                  _legendDot(AppTheme.alertRed, 'THREAT'),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          Expanded(
            child: Consumer<DashboardProvider>(
              builder: (context, provider, child) {
                if (provider.logs.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.show_chart, color: Colors.white24, size: 48),
                        const SizedBox(height: 12),
                        const Text('No data yet — run firebase_upload_service.py',
                            style: TextStyle(color: Colors.white38, fontSize: 13)),
                      ],
                    ),
                  );
                }

                final reversedLogs = provider.logs.take(50).toList().reversed.toList();

                // Build scatter spots colored by classification
                final List<ScatterSpot> spots = [];
                for (int i = 0; i < reversedLogs.length; i++) {
                  final log = reversedLogs[i];
                  final yVal = (_modbusCodeToIndex[log.modbusCommand] ?? 0).toDouble();
                  
                  Color dotColor;
                  final cl = log.classification ?? (log.isAnomalous ? 'THREAT' : 'SAFE');
                  switch (cl) {
                    case 'SAFE':
                      dotColor = AppTheme.safeGreen;
                      break;
                    case 'SUSPICIOUS':
                      dotColor = Colors.amber;
                      break;
                    case 'THREAT':
                      dotColor = AppTheme.alertRed;
                      break;
                    default:
                      dotColor = Colors.white54;
                  }

                  spots.add(ScatterSpot(
                    i.toDouble(),
                    yVal,
                    dotPainter: FlDotCirclePainter(
                      radius: 6,
                      color: dotColor,
                      strokeWidth: 1,
                      strokeColor: dotColor.withOpacity(0.4),
                    ),
                  ));
                }

                return ScatterChart(
                  ScatterChartData(
                    scatterSpots: spots,
                    gridData: FlGridData(
                      show: true,
                      drawVerticalLine: false,
                      getDrawingHorizontalLine: (value) => FlLine(color: Colors.white10, strokeWidth: 1),
                    ),
                    borderData: FlBorderData(show: false),
                    minY: 0,
                    maxY: 29,
                    titlesData: FlTitlesData(
                      show: true,
                      topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                      rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                      leftTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 50,
                          interval: 5,
                          getTitlesWidget: (value, meta) {
                            final idx = value.toInt();
                            if (idx > 0 && idx <= 28 && idx % 5 == 0) {
                              return Text(
                                'MB${idx.toString().padLeft(3, '0')}',
                                style: const TextStyle(color: Colors.white54, fontSize: 10),
                              );
                            }
                            return const SizedBox.shrink();
                          },
                        ),
                      ),
                      bottomTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 22,
                          interval: 5,
                          getTitlesWidget: (value, meta) {
                            int index = value.toInt();
                            if (index >= 0 && index < reversedLogs.length) {
                              return Padding(
                                padding: const EdgeInsets.only(top: 8.0),
                                child: Text(
                                  DateFormat('HH:mm:ss').format(reversedLogs[index].timestamp),
                                  style: const TextStyle(color: Colors.white54, fontSize: 10),
                                ),
                              );
                            }
                            return const SizedBox.shrink();
                          },
                        ),
                      ),
                    ),
                    scatterTouchData: ScatterTouchData(
                      enabled: true,
                      touchTooltipData: ScatterTouchTooltipData(
                        getTooltipItems: (ScatterSpot spot) {
                          final idx = spot.x.toInt();
                          if (idx >= 0 && idx < reversedLogs.length) {
                            final log = reversedLogs[idx];
                            final cl = log.classification ?? (log.isAnomalous ? 'THREAT' : 'SAFE');
                            return ScatterTooltipItem(
                              '${log.modbusCommand} [$cl]\n'
                              '${log.registerName ?? ''}\n'
                              '${DateFormat('HH:mm:ss').format(log.timestamp)}',
                              textStyle: const TextStyle(color: Colors.white, fontSize: 11),
                            );
                          }
                          return ScatterTooltipItem('');
                        },
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      children: [
        Container(width: 10, height: 10, decoration: BoxDecoration(shape: BoxShape.circle, color: color)),
        const SizedBox(width: 6),
        Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
      ],
    );
  }

  Widget _buildModbusLogs(BuildContext context) {
    return GlassContainer(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('MODBUS TELEMETRY LOG', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 16),
          Expanded(
            child: Consumer<DashboardProvider>(
              builder: (context, provider, child) {
                if (provider.logs.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.dns_outlined, color: Colors.white24, size: 48),
                        const SizedBox(height: 12),
                        const Text(
                          'No telemetry received.\nRun firebase_upload_service.py\nto start streaming.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Colors.white38, fontSize: 13),
                        ),
                      ],
                    ),
                  );
                }
                return ListView.builder(
                  itemCount: provider.logs.length,
                  itemBuilder: (context, index) {
                    final log = provider.logs[index];
                    final isAnomalous = log.isAnomalous;
                    final cl = log.classification ?? (isAnomalous ? 'THREAT' : 'SAFE');
                    final clColor = cl == 'SAFE'
                        ? AppTheme.safeGreen
                        : cl == 'SUSPICIOUS'
                            ? Colors.amber
                            : AppTheme.alertRed;

                    return Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: isAnomalous ? AppTheme.alertRed.withOpacity(0.1) : Colors.white.withOpacity(0.05),
                        borderRadius: BorderRadius.circular(8),
                        border: Border(left: BorderSide(color: clColor, width: 4)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                DateFormat('HH:mm:ss').format(log.timestamp),
                                style: const TextStyle(color: Colors.grey, fontSize: 12),
                              ),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                decoration: BoxDecoration(
                                  color: clColor.withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(
                                  cl,
                                  style: TextStyle(
                                    color: clColor,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 11,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Row(
                            children: [
                              Text(
                                log.modbusCommand,
                                style: const TextStyle(
                                  color: AppTheme.accentNeonCyan,
                                  fontFamily: 'monospace',
                                  fontWeight: FontWeight.bold,
                                  fontSize: 14,
                                ),
                              ),
                              if (log.functionCode != null) ...[
                                const SizedBox(width: 8),
                                Text(
                                  log.functionCode!,
                                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                                ),
                              ],
                            ],
                          ),
                          if (log.registerName != null) ...[
                            const SizedBox(height: 2),
                            Text(
                              log.registerName!,
                              style: const TextStyle(color: Colors.white38, fontSize: 11),
                            ),
                          ],
                          if (log.reasoning != null) ...[
                            const SizedBox(height: 4),
                            Text(
                              log.reasoning!,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(color: Colors.white30, fontSize: 11, fontStyle: FontStyle.italic),
                            ),
                          ],
                        ],
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildParametersTab(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final int crossAxisCount = screenWidth > 1200 ? 3 : (screenWidth > 800 ? 2 : 1);
    final double childAspectRatio = screenWidth > 800 ? 3.0 : 2.5;

    return Consumer<DashboardProvider>(
      builder: (context, provider, child) {
        if (provider.logs.isEmpty || provider.logs.first.plantState.isEmpty) {
          return GlassContainer(
            child: const Center(
              child: Text(
                'Awaiting full plant state telemtry...',
                style: TextStyle(color: Colors.white54, fontSize: 16),
              ),
            ),
          );
        }

        final state = provider.logs.first.plantState;
        
        return GridView.builder(
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            childAspectRatio: childAspectRatio,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
          ),
          itemCount: state.keys.length,
          itemBuilder: (context, index) {
            final key = state.keys.elementAt(index);
            final value = state[key];
            
            // Try to make the key more readable
            final displayKey = key.replaceAll('_', ' ').toUpperCase();
            
            return GlassContainer(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    displayKey,
                    style: const TextStyle(color: Colors.white54, fontSize: 11),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value is double ? value.toStringAsFixed(2) : value.toString(),
                    style: const TextStyle(
                      color: AppTheme.accentNeonBlue,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildTrendGraphsTab(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final int crossAxisCount = screenWidth > 1200 ? 3 : (screenWidth > 800 ? 2 : 1);
    final double childAspectRatio = screenWidth > 800 ? 1.5 : 1.2;

    return Consumer<DashboardProvider>(
      builder: (context, provider, child) {
        if (provider.logs.isEmpty || provider.logs.first.plantState.isEmpty) {
          return GlassContainer(
            child: const Center(
              child: Text(
                'Waiting for trend data...',
                style: TextStyle(color: Colors.white54, fontSize: 16),
              ),
            ),
          );
        }

        final List<String> importantKeys = [
          'flow_rate_L_min',
          'distribution_pressure_PSI',
          'chlorine_residual_mg_L',
          'turbidity_treated_NTU',
          'tank_level_pct'
        ];
        
        final stateKeys = provider.logs.first.plantState.keys
            .where((key) => importantKeys.contains(key))
            .toList();
            
        final reversedLogs = provider.logs.take(50).toList().reversed.toList();

        return GridView.builder(
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            childAspectRatio: childAspectRatio,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
          ),
          itemCount: stateKeys.length,
          itemBuilder: (context, index) {
            final key = stateKeys[index];
            final displayKey = key.replaceAll('_', ' ').toUpperCase();

            // Extract min/max to scale the individual chart
            double minY = double.infinity;
            double maxY = double.negativeInfinity;
            final spots = <FlSpot>[];
            
            for (int i = 0; i < reversedLogs.length; i++) {
              final val = reversedLogs[i].plantState[key];
              final double numericVal = (val is num) ? val.toDouble() : 0.0;
              if (numericVal < minY) minY = numericVal;
              if (numericVal > maxY) maxY = numericVal;
              spots.add(FlSpot(i.toDouble(), numericVal));
            }
            
            if (minY == double.infinity) minY = 0;
            if (maxY == double.negativeInfinity) maxY = 1;
            
            // Add slight padding to Y axis
            final yRange = (maxY - minY).abs();
            final yPadding = yRange == 0 ? 1.0 : yRange * 0.1;

            return GlassContainer(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    displayKey,
                    style: const TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: LineChart(
                      LineChartData(
                        gridData: FlGridData(
                          show: true, 
                          drawVerticalLine: false, 
                          getDrawingHorizontalLine: (value) => FlLine(color: Colors.white10, strokeWidth: 1)
                        ),
                        borderData: FlBorderData(show: false),
                        titlesData: const FlTitlesData(
                           leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                           rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                           topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                           bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                        ),
                        minY: minY - yPadding,
                        maxY: maxY + yPadding,
                        lineBarsData: [
                          LineChartBarData(
                            spots: spots,
                            isCurved: true,
                            color: AppTheme.accentNeonCyan,
                            barWidth: 2,
                            isStrokeCapRound: true,
                            dotData: const FlDotData(show: false),
                            belowBarData: BarAreaData(
                              show: true,
                              color: AppTheme.accentNeonCyan.withOpacity(0.1),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}
