import 'package:flutter_test/flutter_test.dart';
import 'package:scada_dashboard/main.dart';

void main() {
  testWidgets('Dashboard startup test', (WidgetTester tester) async {
    await tester.pumpWidget(const ScadaDashboardApp());
    expect(find.text('SCADA Edge AI Security Dashboard'), findsOneWidget);
  });
}
